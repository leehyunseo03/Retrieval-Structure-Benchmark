import os
import json
import random
from typing import List, Set
from dataclasses import dataclass
from tqdm import tqdm
from dotenv import load_dotenv

# LlamaIndex Core
from llama_index.core import (
    Document,
    Settings,
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# =========================================================
# [1. 설정 영역]
# =========================================================

load_dotenv()

# 경로 설정
BASE_DIR = os.environ.get('QASPER_DIR')
PDF_DIR = os.path.join(BASE_DIR, "qasper_pdfs")
QASPER_JSON_PATH = os.path.join(BASE_DIR, "qasper", "qasper-dev-v0.3.json")

# 모델 설정
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# [최적화 1] 청크 전략 수정
Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128)

# 평가 설정
TOP_K_LIST = [1, 3, 5, 10]
LIMIT_PDFS = 30  # 테스트용 개수 제한 (0이면 전체)

# =========================================================
# [2. 데이터 로더] QASPER JSON 파싱
# =========================================================

@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]
    positive_doc_ids: Set[str]

def load_qasper_data(json_path: str) -> List[QAExample]:
    if not os.path.exists(json_path):
        print(f"❌ 파일을 찾을 수 없습니다: {json_path}")
        return []

    print(f"📖 QASPER 데이터 파싱 중...")
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    qa_list = []
    for paper_id, content in raw_data.items():
        qas = content.get("qas", [])
        for qa in qas:
            extracted_answers = []
            for ans_obj in qa.get("answers", []):
                ans_data = ans_obj.get("answer", {})
                if ans_data.get("unanswerable", False): continue 
                
                if ans_data.get("extractive_spans"):
                    extracted_answers.extend(ans_data["extractive_spans"])
                elif ans_data.get("free_form_answer"):
                    extracted_answers.append(ans_data["free_form_answer"])
                elif ans_data.get("yes_no") is not None:
                    extracted_answers.append(str(ans_data["yes_no"]))
            
            if extracted_answers:
                qa_list.append(QAExample(
                    qid=qa.get("question_id"),
                    question=qa.get("question"),
                    answers=extracted_answers,
                    positive_doc_ids={str(paper_id)}
                ))
    return qa_list

# =========================================================
# [3. 문서 로더] 메타데이터 주입 (Strategy 1)
# =========================================================

class OptimizedPDFLoader:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir

    def load_specific_documents(self, target_doc_ids: Set[str]) -> List[Document]:
        all_docs = []
        available_files = {f.replace('.pdf', ''): f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')}
        
        found_ids = [tid for tid in target_doc_ids if tid in available_files]
        if not found_ids: return []

        print(f"📂 PDF 로딩 및 메타데이터 주입 ({len(found_ids)}개)...")
        for doc_id in tqdm(found_ids, desc="Processing PDFs"):
            filename = available_files[doc_id]
            filepath = os.path.join(self.pdf_dir, filename)
            
            try:
                # 파일 로드
                text_docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
                
                for d in text_docs:
                    d.metadata["doc_id"] = doc_id
                    d.metadata["file_name"] = filename
                    
                    # [최적화 1: 메타데이터 주입]
                    # 텍스트 맨 앞에 논문 ID를 명시하여 검색 혼동 방지
                    header = f"[Paper ID: {doc_id}]\n"
                    new_text = header + d.get_content()
                    d.set_content(new_text)

                all_docs.extend(text_docs)
            except Exception as e:
                print(f"   [Error] {filename}: {e}")
                
        return all_docs

# =========================================================
# [4. 메인 평가 로직] Hybrid Search + Re-ranking
# =========================================================

def evaluate_system():
    
    print("\n" + "="*60)
    print("🚀 QASPER Advanced RAG Evaluation (Corrected)")
    print("   1. Metadata Injection (Contextual Chunking)")
    print("   2. Hybrid Search (Vector + BM25)")
    print("   3. Re-ranking (Cross-Encoder)")
    print("="*60)

    # 1. 데이터 준비
    full_qa_list = load_qasper_data(QASPER_JSON_PATH)
    if not full_qa_list or not os.path.exists(PDF_DIR): return

    available_pdfs_ids = set(f.replace('.pdf', '') for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    needed_ids = set()
    for qa in full_qa_list: needed_ids.update(qa.positive_doc_ids)
    valid_ids = list(needed_ids.intersection(available_pdfs_ids))

    if LIMIT_PDFS > 0 and len(valid_ids) > LIMIT_PDFS:
        target_doc_ids = set(random.sample(valid_ids, LIMIT_PDFS))
    else:
        target_doc_ids = set(valid_ids)

    print(f"📊 평가 규모: {len(target_doc_ids)}개 논문 (Total QA: {len(full_qa_list)})")

    # 2. 문서 로드 및 인덱싱
    loader = OptimizedPDFLoader(PDF_DIR)
    docs = loader.load_specific_documents(target_doc_ids)
    
    print(f"\n🏗️  Vector Index & BM25 Index 생성 중... (청크 수: {len(docs)})")
    
    # [인덱싱] VectorStoreIndex 사용
    index = VectorStoreIndex.from_documents(docs, show_progress=True)
    
    # 3. [최적화 2] Hybrid Search 구성
    print("🔗 Hybrid Retriever 구성 중 (Vector + BM25)...")
    
    # (1) Vector Retriever
    vector_retriever = index.as_retriever(similarity_top_k=20) 
    
    # (2) BM25 Retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=index.docstore.docs.values(),
        similarity_top_k=20,
        language="english"
    )
    
    # (3) Fusion
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=True,
        similarity_top_k=20
    )

    # 4. [최적화 3] Re-ranking 구성
    print("🎯 Re-ranker (Cross-Encoder) 로딩 중...")
    # 'cross-encoder/ms-marco-MiniLM-L-6-v2' 모델 사용 (자동 다운로드됨)
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=max(TOP_K_LIST) 
    )

    # 5. 평가 루프
    eval_qa_list = [qa for qa in full_qa_list if not qa.positive_doc_ids.isdisjoint(target_doc_ids)]
    print(f"\n🔎 평가 시작 (총 {len(eval_qa_list)}개 질문)")
    
    metrics = {f"recall@{k}": 0.0 for k in TOP_K_LIST}
    metrics.update({f"mrr@{k}": 0.0 for k in TOP_K_LIST})

    for i, ex in enumerate(tqdm(eval_qa_list, desc="Evaluating")):
        try:
            # Step A: Hybrid Retrieve -> Top 20 Candidates
            initial_nodes = hybrid_retriever.retrieve(ex.question)
            
            # Step B: Re-ranking -> Top N Final
            reranked_nodes = reranker.postprocess_nodes(
                initial_nodes, 
                query_str=ex.question
            )
            
            final_nodes = reranked_nodes
            
        except Exception as e:
            print(f"Error retrieving: {e}")
            final_nodes = []
        
        retrieved_doc_ids = [node.metadata.get("doc_id", "") for node in final_nodes]
        gt_set = ex.positive_doc_ids

        # 디버깅 (첫 3개)
        if i < 3:
            tqdm.write(f"\n[Q] {ex.question}")
            tqdm.write(f"   Target: {list(gt_set)}")
            tqdm.write(f"   Pred:   {retrieved_doc_ids[:5]}")
            hit = any(d in gt_set for d in retrieved_doc_ids[:5])
            tqdm.write(f"   -> {'✅ HIT' if hit else '❌ MISS'}")

        # Metrics 계산
        for k in TOP_K_LIST:
            current_top_k = retrieved_doc_ids[:min(k, len(retrieved_doc_ids))]
            
            if any(did in gt_set for did in current_top_k):
                metrics[f"recall@{k}"] += 1.0
            
            for rank, did in enumerate(current_top_k, start=1):
                if did in gt_set:
                    metrics[f"mrr@{k}"] += 1.0 / rank
                    break

    # 6. 최종 결과
    count = len(eval_qa_list)
    print("\n" + "="*50)
    print(f"📈 최종 Advanced 평가 결과 (Samples: {count})")
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