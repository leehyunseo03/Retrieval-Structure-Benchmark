import os
import json
import shutil
import random
import time
from typing import List, Set, Dict
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image

# 환경 변수 로드
from dotenv import load_dotenv

# LlamaIndex 관련 임포트
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    PropertyGraphIndex
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from google import genai

# =========================================================
# [1. 설정 영역] 경로 및 API 키
# =========================================================

load_dotenv()

# API 키 확인
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  [Warning] OPENAI_API_KEY가 설정되지 않았습니다.")
    
# 경로 설정 (다운로드 스크립트와 동일한 구조라고 가정)
BASE_DIR = os.environ.get('QASPER_DIR')
PDF_DIR = os.path.join(BASE_DIR, "qasper_pdfs")
QASPER_JSON_PATH = os.path.join(BASE_DIR, "qasper", "qasper-dev-v0.3.json")

# 모델 설정
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Gemini Vision (이미지 캡션용 - 필요시 사용)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
VISION_MODEL_ID = 'gemini-2.0-flash'

# 평가 설정
TOP_K_LIST = [1, 3, 5]  # 측정할 Top-K 지표
LIMIT_PDFS = 30         # 테스트 속도를 위해 사용할 PDF 개수 제한 (0이면 전체, 추천: 20~50)

# =========================================================
# [2. 데이터 로더] QASPER JSON 파싱
# =========================================================

@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]
    positive_doc_ids: Set[str] # 정답 논문 ID (QASPER는 1개)

def load_qasper_data(json_path: str) -> List[QAExample]:
    """QASPER JSON을 읽어 평가용 객체 리스트로 변환"""
    if not os.path.exists(json_path):
        print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
        return []

    print(f"📖 QASPER 데이터 파싱 중: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    qa_list = []
    
    # raw_data 구조: { "PAPER_ID": { "qas": [...] }, ... }
    for paper_id, content in raw_data.items():
        qas = content.get("qas", [])
        
        for qa in qas:
            # 답변 추출 로직
            extracted_answers = []
            
            # QASPER는 여러 답변자가 있을 수 있음
            for ans_obj in qa.get("answers", []):
                ans_data = ans_obj.get("answer", {})
                
                # '답변 없음(Unanswerable)' 체크
                if ans_data.get("unanswerable", False):
                    continue 

                # 1. 추출형 (Extractive)
                if ans_data.get("extractive_spans"):
                    extracted_answers.extend(ans_data["extractive_spans"])
                # 2. 요약형 (Abstractive)
                elif ans_data.get("free_form_answer"):
                    extracted_answers.append(ans_data["free_form_answer"])
                # 3. Yes/No
                elif ans_data.get("yes_no") is not None:
                    extracted_answers.append(str(ans_data["yes_no"]))
            
            # 유효한 답변이 있는 질문만 추가
            if extracted_answers:
                qa_list.append(QAExample(
                    qid=qa.get("question_id"),
                    question=qa.get("question"),
                    answers=extracted_answers,
                    positive_doc_ids={str(paper_id)} # 해당 논문 ID가 정답
                ))

    return qa_list

# =========================================================
# [3. 문서 로더] PDF 로드 및 전처리
# =========================================================

class SmartPDFLoader:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def load_specific_documents(self, target_doc_ids: Set[str]) -> List[Document]:
        all_docs = []
        
        # 실제 존재하는 파일만 필터링
        available_files = {f.replace('.pdf', ''): f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')}
        
        found_ids = []
        for tid in target_doc_ids:
            if tid in available_files:
                found_ids.append(tid)
        
        if not found_ids:
            return []

        print(f"📂 PDF 파일 로딩 시작 ({len(found_ids)}개)...")
        for doc_id in tqdm(found_ids, desc="Loading PDFs"):
            filename = available_files[doc_id]
            filepath = os.path.join(self.pdf_dir, filename)
            
            try:
                # 1. 텍스트 로드 (SimpleDirectoryReader)
                # filename_as_id=True를 쓰면 doc_id 관리가 편함
                text_docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
                
                for d in text_docs:
                    # 메타데이터에 정답 매칭을 위한 doc_id 주입 (필수)
                    d.metadata["doc_id"] = doc_id 
                    d.metadata["file_name"] = filename
                
                all_docs.extend(text_docs)
                
                # 이미지 캡션 로직은 속도 관계상 생략하거나 주석 처리
               
                
            except Exception as e:
                print(f"   [Error] {filename} 로드 실패: {e}")
                
        return all_docs

# =========================================================
# [4. 메인 평가 로직]
# =========================================================

def evaluate_system():
    print("\n" + "="*50)
    print("🚀 QASPER RAG 성능 평가 (Document Retrieval)")
    print("="*50)

    # 1. QA 데이터 로드
    full_qa_list = load_qasper_data(QASPER_JSON_PATH)
    if not full_qa_list:
        return

    # 2. 사용 가능한 PDF 확인
    if not os.path.exists(PDF_DIR):
        print(f"❌ PDF 폴더가 없습니다: {PDF_DIR}")
        return
        
    available_pdfs_ids = set(f.replace('.pdf', '') for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    
    # QA 데이터에 있는 ID와 실제 PDF가 있는 ID의 교집합 찾기
    needed_ids = set()
    for qa in full_qa_list:
        needed_ids.update(qa.positive_doc_ids)
        
    valid_ids = list(needed_ids.intersection(available_pdfs_ids))
    
    if not valid_ids:
        print("❌ 매칭되는 PDF 파일이 하나도 없습니다. 파일명(PaperID)을 확인하세요.")
        return

    # 3. 평가 범위 설정 (LIMIT_PDFS)
    # 랜덤으로 일부 논문만 선택하여 인덱싱 (전체는 너무 오래 걸림)
    if LIMIT_PDFS > 0 and len(valid_ids) > LIMIT_PDFS:
        target_doc_ids = set(random.sample(valid_ids, LIMIT_PDFS))
    else:
        target_doc_ids = set(valid_ids)

    print(f"📊 통계:")
    print(f" - 전체 질문 수: {len(full_qa_list)}")
    print(f" - PDF 보유 논문 수: {len(available_pdfs_ids)}")
    print(f" - 🎯 이번 평가에 사용할 논문 수: {len(target_doc_ids)}개")

    # 4. 문서 로딩 및 인덱싱
    loader = SmartPDFLoader(PDF_DIR)
    docs = loader.load_specific_documents(target_doc_ids)
    
    print(f"\n🏗️  Index(Knowledge Graph) 생성 중... (청크 수: {len(docs)})")
    # PropertyGraphIndex 생성 (시간이 좀 걸립니다)
    index = PropertyGraphIndex.from_documents(
        docs,
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        kg_extractors=[SimpleLLMPathExtractor(llm=Settings.llm, max_paths_per_chunk=5)],
        show_progress=False
    )
    
    retriever = index.as_retriever(include_text=True, similarity_top_k=max(TOP_K_LIST))

    # 5. 평가 대상 질문 필터링
    # 우리가 인덱싱한 문서(target_doc_ids)에 대한 질문만 평가해야 함
    eval_qa_list = [qa for qa in full_qa_list if not qa.positive_doc_ids.isdisjoint(target_doc_ids)]
    
    print(f"\n🔎 평가 시작 (총 {len(eval_qa_list)}개 질문)")
    
    # 메트릭 초기화
    metrics = {f"recall@{k}": 0.0 for k in TOP_K_LIST}
    metrics.update({f"mrr@{k}": 0.0 for k in TOP_K_LIST})

    # 6. 검색 및 채점 루프
    for i, ex in enumerate(tqdm(eval_qa_list, desc="Evaluating")):
        try:
            nodes = retriever.retrieve(ex.question)
        except Exception as e:
            nodes = []
        
        # 검색된 문서들의 ID 추출 (중복 제거 없이 순위 유지)
        retrieved_doc_ids = [node.metadata.get("doc_id", "") for node in nodes]
        
        # 정답셋 (Paper ID)
        gt_set = ex.positive_doc_ids

        # --- [디버깅 출력: 처음 3개만 자세히 보기] ---
        if i < 3:
            tqdm.write(f"\n[Q] {ex.question}")
            tqdm.write(f"   (정답 ID: {list(gt_set)})")
            tqdm.write(f"   (검색된 ID: {retrieved_doc_ids[:5]})")
            hit = any(d in gt_set for d in retrieved_doc_ids[:5])
            tqdm.write(f"   -> {'✅ HIT' if hit else '❌ MISS'}")
        # ----------------------------------------

        for k in TOP_K_LIST:
            current_top_k = retrieved_doc_ids[:k]
            
            # Recall Check (하나라도 정답 문서가 있으면 성공)
            if any(did in gt_set for did in current_top_k):
                metrics[f"recall@{k}"] += 1.0
            
            # MRR Check
            for rank, did in enumerate(current_top_k, start=1):
                if did in gt_set:
                    metrics[f"mrr@{k}"] += 1.0 / rank
                    break

    # 7. 최종 결과 출력
    count = len(eval_qa_list)
    print("\n" + "="*50)
    print(f"📈 최종 평가 결과 (Samples: {count})")
    print("="*50)
    
    if count > 0:
        print(f"{'Metric':<12} | {'Score':<10}")
        print("-" * 25)
        for k in TOP_K_LIST:
            recall = metrics[f'recall@{k}'] / count
            mrr = metrics[f'mrr@{k}'] / count
            print(f"Recall@{k:<2}   | {recall:.4f}")
            print(f"MRR@{k:<2}      | {mrr:.4f}")
    else:
        print("평가된 질문이 없습니다.")

if __name__ == "__main__":
    evaluate_system()