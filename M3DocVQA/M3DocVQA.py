import os
import json
import ujson
import fitz  # PyMuPDF
import time
import random
import shutil
import datetime
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Set
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image

# LlamaIndex 관련 임포트
from llama_index.core import (
    Document,
    VectorStoreIndex,
    Settings,
    StorageContext,
    SimpleDirectoryReader,
    PropertyGraphIndex
)
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from google import genai

# =========================================================
# [설정 영역] API KEY 및 경로
# =========================================================

# 1. API 키 설정
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# 2. 데이터 경로 설정
# M3DocVQA 데이터셋이 있는 폴더 경로
BASE_DIR =  os.environ.get('M3DocVQA_DIR')
PDF_DIR = os.path.join(BASE_DIR, "pdfs_dev") # PDF 파일들이 들어있는 폴더
DEV_QA_PATH = os.path.join(BASE_DIR, "multimodalqa", "MMQA_dev.jsonl") # 질문지 파일

# 3. 모델 설정
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

client = genai.Client(api_key=GOOGLE_API_KEY)
vision_model_id = 'gemini-2.0-flash'

# 4. 평가 설정
TOP_K_LIST = [1, 3, 5, 10]
LIMIT_PDFS = 30  # 테스트를 위해 처리할 PDF 개수 제한 (전체는 시간이 오래 걸림). 0이면 전체 실행.

# =========================================================
# [Part 1] 데이터 로더 및 전처리 (M3DocVQA 형식)
# =========================================================

@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]
    positive_doc_ids: Set[str]

def load_mmqa_data(jsonl_path: str) -> List[QAExample]:
    qa_list = []
    if not os.path.exists(jsonl_path):
        print(f"[Error] QA 파일을 찾을 수 없습니다: {jsonl_path}")
        return []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            ex = ujson.loads(line)

            qid = ex.get("qid") or str(ex.get("id"))
            question = ex["question"]
            
            # 정답 텍스트 추출
            answers = [a["answer"] if isinstance(a, dict) else a for a in ex.get("answers", [])]
            
            # [핵심 수정] supporting_context에서 doc_id 추출
            positive_doc_ids = set()
            
            # 1. supporting_context 확인
            if "supporting_context" in ex:
                for ctx in ex["supporting_context"]:
                    doc_id = ctx.get("doc_id")
                    if doc_id:
                        positive_doc_ids.add(str(doc_id))
            
            # 2. metadata 내 image_doc_ids 등 확인 (보조)
            if "metadata" in ex:
                meta = ex["metadata"]
                if "image_doc_ids" in meta:
                    for img_id in meta["image_doc_ids"]:
                        # 질문과 연관된 이미지가 명확하면 추가할 수도 있으나, 
                        # 보통 supporting_context가 정답 근거임. 여기선 supporting_context가 비었을 때만 고려
                        if not positive_doc_ids: 
                            pass # 여기서는 일단 supporting_context만 신뢰함

            if positive_doc_ids:
                qa_list.append(QAExample(qid, question, answers, positive_doc_ids))
            
    return qa_list

# =========================================================
# [Part 2] 사용자 커스텀 인덱싱 (MongoDB 제거, Gemini 캡션 포함)
# =========================================================

def generate_image_caption(image_path: str) -> str:
    """Google Gemini를 사용하여 이미지 캡션을 생성합니다."""
    try:
        img = Image.open(image_path)
        if img.width < 100 or img.height < 100:
            return "식별 불가능한 작은 이미지 또는 아이콘"

        prompt = (
            "Describe this image in detail for a search engine. "
            "If it's a chart, include the numbers. If it's text, summarize it."
        )
        time.sleep(1.5) # Rate Limit 고려
        response = client.models.generate_content(model=vision_model_id, contents=[prompt, img])
        return response.text if response.text else "설명 생성 실패"
    except Exception as e:
        # print(f"   [Caption Error] {str(e)[:50]}...")
        return "이미지 처리를 할 수 없습니다."

class SmartPDFLoader:
    def __init__(self, pdf_dir: str, output_img_dir: str = "temp_images"):
        self.pdf_dir = pdf_dir
        self.output_img_dir = output_img_dir
        if os.path.exists(self.output_img_dir):
            shutil.rmtree(self.output_img_dir)
        os.makedirs(self.output_img_dir, exist_ok=True)

    def load_specific_documents(self, target_doc_ids: Set[str]) -> List[Document]:
        all_docs = []
        
        # 파일명 매칭
        available_files = {f.split('.')[0]: f for f in os.listdir(self.pdf_dir) if f.lower().endswith(".pdf")}
        
        found_files = []
        for tid in target_doc_ids:
            if tid in available_files:
                found_files.append(available_files[tid])
        
        if not found_files:
            return []

        for filename in tqdm(found_files, desc="📂 문서 처리 중", unit="file"):
            filepath = os.path.join(self.pdf_dir, filename)
            doc_id = filename.split('.')[0] # 확장자 뺀 파일명을 doc_id로 사용
            
            # 1. 텍스트
            try:
                text_docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
                for d in text_docs:
                    d.metadata["doc_id"] = doc_id
                all_docs.extend(text_docs)
            except:
                pass
            """
            # 2. 이미지
            try:
                fitz_doc = fitz.open(filepath)
                for page_idx, page in enumerate(fitz_doc):
                    image_list = page.get_images(full=True)
                    if not image_list: continue

                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            base_image = fitz_doc.extract_image(xref)
                            saved_path = os.path.join(self.output_img_dir, f"{doc_id}_p{page_idx}_{img_idx}.{base_image['ext']}")
                            with open(saved_path, "wb") as f:
                                f.write(base_image["image"])
                            
                            caption = generate_image_caption(saved_path)
                            img_doc = Document(
                                text=f"[이미지 설명]\n파일: {filename} p.{page_idx+1}\n내용: {caption}",
                                metadata={"doc_id": doc_id, "page_label": str(page_idx + 1), "type": "image"}
                            )
                            all_docs.append(img_doc)
                        except:
                            continue
                fitz_doc.close()
            except:
                pass
            """
        return all_docs
    
# =========================================================
# [Part 3] 실행 및 평가 (MMQA 로직 적용)
# =========================================================

def evaluate_system():
    print("\n" + "="*40)
    print("🚀 MMQA 평가 시스템 시작 (Quiet Mode)")
    print("="*40)

    # 1. QA 데이터 로드
    full_qa_list = load_mmqa_data(DEV_QA_PATH)
    if not full_qa_list: return

    # 2. 필요한 문서 ID 추출
    needed_doc_ids = set()
    for qa in full_qa_list:
        needed_doc_ids.update(qa.positive_doc_ids)
            
    # PDF 폴더 스캔
    available_pdfs = set(f.split('.')[0] for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    
    # 교집합 확인
    valid_doc_ids = list(needed_doc_ids.intersection(available_pdfs))
    
    if not valid_doc_ids:
        print("❌ 질문지의 doc_id와 일치하는 PDF 파일이 없습니다.")
        print(f"예시 QA doc_id: {list(needed_doc_ids)[:3]}")
        print(f"예시 PDF 파일명: {list(available_pdfs)[:3]}")
        return

    random.shuffle(valid_doc_ids)
    target_doc_ids = set(valid_doc_ids[:LIMIT_PDFS])
    
    print(f"📋 전체 QA: {len(full_qa_list)}개")
    print(f"🎯 매칭된 문서 중 {len(target_doc_ids)}개만 로드하여 평가 진행 (LIMIT)")

    # 3. 문서 로드
    loader = SmartPDFLoader(PDF_DIR)
    docs = loader.load_specific_documents(target_doc_ids)
    
    # 4. 인덱스 생성
    print(f"🏗️  Knowledge Graph 생성 중... ({len(docs)})")
    index = PropertyGraphIndex.from_documents(
        docs,
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        kg_extractors=[SimpleLLMPathExtractor(llm=Settings.llm, max_paths_per_chunk=5)],
        show_progress=False 
    )
    
    retriever = index.as_retriever(include_text=True, similarity_top_k=max(TOP_K_LIST))

    # 5. 평가 대상 필터링 (로드한 문서에 대한 질문만)
    filtered_qa_list = []
    for qa in full_qa_list:
        # 질문의 정답 문서(positive_doc_ids) 중 하나라도 로드된 문서(target_doc_ids)에 있으면 평가
        if not qa.positive_doc_ids.isdisjoint(target_doc_ids):
            filtered_qa_list.append(qa)
            
    print(f"🔎 관련 질문 {len(filtered_qa_list)}개에 대해 평가 진행")

    # 6. 평가 루프
    metrics = {f"recall@{k}": 0.0 for k in TOP_K_LIST}
    metrics.update({f"mrr@{k}": 0.0 for k in TOP_K_LIST})
    
    for i, ex in enumerate(tqdm(filtered_qa_list, desc="📝 평가 진행 중", unit="Q")):
        try:
            nodes = retriever.retrieve(ex.question)
        except:
            nodes = []

        retrieved_docs = []
        retrieved_details = []
        retrieved_doc_ids = []

        # 검색된 문서 ID 리스트 추출
        retrieved_doc_ids = []
        for node in nodes:
            r_doc_id = node.metadata.get("doc_id", "")
            retrieved_doc_ids.append(r_doc_id)

            content_preview = node.text[:100].replace('\n', ' ') + "..."
            retrieved_details.append(f"[{r_doc_id}] {content_preview}")

        # 정답셋 (문서 ID)
        gt_set = ex.positive_doc_ids

        # --- [상세 결과 출력 부분] ---
        tqdm.write("\n" + "-"*60)
        tqdm.write(f"📌 [Question #{i+1}] {ex.question}")
        tqdm.write(f"✅ 정답(GT Doc IDs): {list(gt_set)}")
        tqdm.write(f"💬 정답 텍스트(참고용): {ex.answers}")
        tqdm.write(f"🔍 예측(Retrieved Top-{max(TOP_K_LIST)}):")
        for rank, detail in enumerate(retrieved_details, 1):
            tqdm.write(f"   {rank}. {detail}")
        
        # 결과 판정 (Top-5 기준 Hit 여부 출력)
        hit_check = any(d in gt_set for d in retrieved_doc_ids[:5])
        tqdm.write(f"🎯 결과: {'⭕ Hit' if hit_check else '❌ Miss'}")
        tqdm.write("-" * 60)
        # ---------------------------

        for k in TOP_K_LIST:
            current_top_k = retrieved_doc_ids[:k]
            
            # Recall Check: Top-K 안에 정답 문서 ID가 하나라도 있는지
            is_hit = any(did in gt_set for did in current_top_k)
            if is_hit: 
                metrics[f"recall@{k}"] += 1.0
            
            # MRR Check
            for rank, did in enumerate(current_top_k, start=1):
                if did in gt_set:
                    metrics[f"mrr@{k}"] += 1.0 / rank
                    break

    # 7. 결과 출력
    count = len(filtered_qa_list)
    print("\n" + "="*30)
    print(f"📊 최종 결과 (Samples: {count})")
    print("="*30)
    
    if count > 0:
        for k in TOP_K_LIST:
            recall = metrics[f'recall@{k}']/count
            mrr = metrics[f'mrr@{k}']/count
            print(f"K={k:<2} | Recall: {recall:.4f} | MRR: {mrr:.4f}")
    
    if os.path.exists("temp_images"):
        shutil.rmtree("temp_images")

if __name__ == "__main__":
    evaluate_system() 