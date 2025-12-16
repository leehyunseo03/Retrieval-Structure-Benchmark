import os
import json
import random
import shutil
from typing import List, Set, Dict
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    PropertyGraphIndex
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# =========================================================
# [1. 설정 영역]
# =========================================================

load_dotenv()

# 경로 설정 (다운로드 스크립트와 동일해야 함)
BASE_DIR = os.environ.get('FinanceBench_DIR')
PDF_DIR = os.path.join(BASE_DIR, "financebench_pdfs")
FB_JSON_PATH = os.path.join(BASE_DIR, "financebench", "financebench_data.json")

# 모델 설정
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 평가 설정
TOP_K_LIST = [1, 3, 5, 10]

# ⚠️ 주의: FinanceBench 문서는 페이지가 매우 많습니다 (평균 100p 이상).
# 테스트 시에는 2~5개 정도로 작게 설정하는 것을 강력 추천합니다.
LIMIT_PDFS = 5 

# =========================================================
# [2. 데이터 로더] JSON 파싱
# =========================================================

@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]      
    positive_doc_ids: Set[str] # 정답 문서 파일명 (ID)

def load_financebench_data(json_path: str) -> List[QAExample]:
    """JSON 파일을 읽어 QAExample 리스트로 변환"""
    if not os.path.exists(json_path):
        print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
        return []

    print(f"📖 FinanceBench 데이터 로딩 중...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_list = []
    # 데이터 구조: { "Safe_Filename": { "qas": [...] }, ... }
    for doc_filename, content in data.items():
        # 해당 문서에 속한 질문들 순회
        for qa in content["qas"]:
            qa_list.append(QAExample(
                qid=str(qa["qid"]),
                question=qa["question"],
                answers=[qa["answer"]], 
                # 이 질문의 정답 문서는 이 JSON Key(파일명) 자체임
                positive_doc_ids={doc_filename} 
            ))
            
    return qa_list

# =========================================================
# [3. 문서 로더] PDF 로드 및 메타데이터 주입
# =========================================================

class SmartPDFLoader:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def load_specific_documents(self, target_doc_ids: Set[str]) -> List[Document]:
        all_docs = []
        
        # 폴더 내 실제 파일 확인 (확장자 제외한 이름을 ID로 매칭)
        # 예: "3M_2018_10K.pdf" -> ID: "3M_2018_10K"
        available_files = {f.replace('.pdf', ''): f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')}
        
        found_ids = []
        for tid in target_doc_ids:
            if tid in available_files:
                found_ids.append(tid)
        
        if not found_ids:
            return []

        print(f"📂 문서 로딩 시작 ({len(found_ids)}개)...")
        # FinanceBench는 파일이 크므로 하나씩 로딩 과정을 보여줍니다.
        for doc_id in tqdm(found_ids, desc="Loading PDFs"):
            filename = available_files[doc_id]
            filepath = os.path.join(self.pdf_dir, filename)
            
            try:
                # 파일 로드
                text_docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
                
                for d in text_docs:
                    # [중요] 메타데이터에 문서 ID(파일명) 주입
                    d.metadata["doc_id"] = doc_id
                    d.metadata["file_name"] = filename
                    
                    # [Tip] 금융 문서는 여러 회사 내용이 섞일 수 있으므로 
                    # 텍스트 앞단에 파일명(회사명+연도)을 명시하면 검색 성능이 오릅니다.
                    # d.text = f"Source Document: {doc_id}\nContent: {d.text}"

                all_docs.extend(text_docs)
            except Exception as e:
                print(f"   [Error] {filename} 로드 실패: {e}")
                
        return all_docs

# =========================================================
# [4. 메인 평가 로직]
# =========================================================

def evaluate_system():
    print("\n" + "="*50)
    print("💰 FinanceBench RAG 성능 평가 (Retrieval)")
    print("="*50)

    # 1. QA 데이터 로드
    full_qa_list = load_financebench_data(FB_JSON_PATH)
    if not full_qa_list:
        return

    # 2. PDF 폴더 확인 및 매칭
    if not os.path.exists(PDF_DIR):
        print(f"❌ PDF 폴더가 없습니다: {PDF_DIR}")
        return
    
    available_pdfs = set(f.replace('.pdf', '') for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    
    # 질문지에 있는 문서 ID들 추출
    needed_ids = set()
    for qa in full_qa_list:
        needed_ids.update(qa.positive_doc_ids)
        
    # 실제 PDF가 존재하는 문서만 유효
    valid_ids = list(needed_ids.intersection(available_pdfs))
    
    if not valid_ids:
        print("❌ 질문 데이터와 매칭되는 PDF 파일이 없습니다.")
        print("   (download_financebench.py 실행 여부와 경로를 확인하세요)")
        return

    # 3. 평가 범위 설정 (LIMIT_PDFS)
    if LIMIT_PDFS > 0 and len(valid_ids) > LIMIT_PDFS:
        target_doc_ids = set(random.sample(valid_ids, LIMIT_PDFS))
    else:
        target_doc_ids = set(valid_ids)
        
    print(f"📊 평가 구성:")
    print(f" - 전체 질문 수: {len(full_qa_list)}")
    print(f" - 보유 PDF 수: {len(available_pdfs)}")
    print(f" - 🎯 이번 평가 대상 문서: {len(target_doc_ids)}개")
    if LIMIT_PDFS > 0:
        print(f"   (주의: LIMIT_PDFS={LIMIT_PDFS} 설정됨)")

    # 4. 인덱싱 (Knowledge Graph)
    loader = SmartPDFLoader(PDF_DIR)
    docs = loader.load_specific_documents(target_doc_ids)
    
    print(f"\n🏗️  Index 생성 중... (총 청크 수: {len(docs)})")
    # 문서 양이 많으므로 show_progress=True 필수
    index = PropertyGraphIndex.from_documents(
        docs,
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        kg_extractors=[SimpleLLMPathExtractor(llm=Settings.llm, max_paths_per_chunk=5)],
        show_progress=True
    )
    
    retriever = index.as_retriever(include_text=True, similarity_top_k=max(TOP_K_LIST))

    # 5. 질문 필터링 (우리가 로드한 문서에 대한 질문만 추리기)
    eval_qa_list = [qa for qa in full_qa_list if not qa.positive_doc_ids.isdisjoint(target_doc_ids)]
    print(f"\n🔎 평가 시작 (총 {len(eval_qa_list)}개 질문)")

    # 6. 평가 루프
    metrics = {f"recall@{k}": 0.0 for k in TOP_K_LIST}
    metrics.update({f"mrr@{k}": 0.0 for k in TOP_K_LIST})

    for i, ex in enumerate(tqdm(eval_qa_list, desc="Evaluating")):
        try:
            nodes = retriever.retrieve(ex.question)
        except Exception:
            nodes = []
        
        # 검색된 문서 ID 추출
        retrieved_ids = [node.metadata.get("doc_id", "") for node in nodes]
        gt_set = ex.positive_doc_ids

        # --- 디버깅용 출력 (첫 번째 질문만) ---
        if i == 0:
            tqdm.write(f"\n[Q sample] {ex.question}")
            tqdm.write(f"   Target Doc: {list(gt_set)}")
            tqdm.write(f"   Retrieved Top-3: {retrieved_ids[:3]}")
            hit = any(d in gt_set for d in retrieved_ids[:3])
            tqdm.write(f"   Result: {'✅ Hit' if hit else '❌ Miss'}\n")
        # ----------------------------------

        for k in TOP_K_LIST:
            current_top_k = retrieved_ids[:k]
            
            # Recall Check
            if any(did in gt_set for did in current_top_k):
                metrics[f"recall@{k}"] += 1.0
            
            # MRR Check
            for rank, did in enumerate(current_top_k, start=1):
                if did in gt_set:
                    metrics[f"mrr@{k}"] += 1.0 / rank
                    break

    # 7. 최종 결과
    count = len(eval_qa_list)
    print("\n" + "="*50)
    print(f"📈 FinanceBench 평가 결과 (Samples: {count})")
    print("="*50)
    
    if count > 0:
        for k in TOP_K_LIST:
            recall = metrics[f'recall@{k}'] / count
            mrr = metrics[f'mrr@{k}'] / count
            print(f"Recall@{k:<2}   | {recall:.4f}")
            print(f"MRR@{k:<2}      | {mrr:.4f}")
    else:
        print("평가할 질문이 없습니다 (문서 매칭 실패 가능성).")

if __name__ == "__main__":
    evaluate_system()