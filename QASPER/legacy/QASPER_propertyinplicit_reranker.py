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
    PropertyGraphIndex, 
    StorageContext
)
# [변경] 느린 LLM 추출기 대신, 빠른 'Implicit' 추출기 사용
from llama_index.core.indices.property_graph import ImplicitPathExtractor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Retrieval & Post-processing
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

# =========================================================
# [1. 설정 영역]
# =========================================================

load_dotenv()

BASE_DIR = os.environ.get('QASPER_DIR')
PDF_DIR = os.path.join(BASE_DIR, "qasper_pdfs")
QASPER_JSON_PATH = os.path.join(BASE_DIR, "qasper", "qasper-dev-v0.3.json")

# 모델 설정
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 청크 설정
parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128)
Settings.node_parser = parser

TOP_K_LIST = [1, 3, 5, 10]
LIMIT_PDFS = 30 

# =========================================================
# [2. 데이터 로더] 
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
# [3. 문서 로더] 메타데이터 주입
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
                text_docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
                
                for d in text_docs:
                    d.metadata["doc_id"] = doc_id
                    d.metadata["file_name"] = filename
                    
                    header = f"[Paper ID: {doc_id}]\n"
                    original_text = d.get_content()
                    d.set_content(header + original_text)

                all_docs.extend(text_docs)
            except Exception as e:
                print(f"   [Error] {filename}: {e}")
                
        return all_docs

# =========================================================
# [4. 메인 평가 로직] FAST PropertyGraph + Hybrid + Rerank
# =========================================================

def evaluate_system():
    print("\n" + "="*70)
    print("🚀 QASPER Fast Graph Evaluation")
    print("   1. Metadata Injection")
    print("   2. Fast Property Graph (Implicit Structure)")
    print("   3. Hybrid Search + Re-ranking")
    print("="*70)

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

    print(f"📊 평가 규모: {len(target_doc_ids)}개 논문")

    # 2. 문서 로드 및 노드 생성
    loader = OptimizedPDFLoader(PDF_DIR)
    docs = loader.load_specific_documents(target_doc_ids)
    
    print("🔨 문서를 노드로 분할 중...")
    nodes = parser.get_nodes_from_documents(docs)

    # 3. Property Graph Index 생성 (Fast Mode)
    print(f"\n🏗️  Property Graph Index 생성 중 (Nodes: {len(nodes)})...")
    
    # [수정됨] LLM 대신 ImplicitPathExtractor 사용
    # 문서의 순서(Next/Prev)와 소속(Parent) 관계만으로 그래프를 만듭니다. (매우 빠름)
    index = PropertyGraphIndex(
        nodes=nodes,
        kg_extractors=[ImplicitPathExtractor()], 
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        show_progress=True
    )
    
    # 4. Hybrid Retriever 구성
    print("🔗 Hybrid Retriever 구성...")
    
    # (A) Graph Retriever
    pg_retriever = index.as_retriever(
        include_text=True, 
        similarity_top_k=20
    )
    
    # (B) BM25 Retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=20,
        language="english"
    )
    
    # (C) Fusion
    hybrid_retriever = QueryFusionRetriever(
        [pg_retriever, bm25_retriever],
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=True,
        similarity_top_k=20
    )

    # 5. Re-ranking
    print("🎯 Re-ranker 로딩 중...")
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2", 
        top_n=max(TOP_K_LIST) 
    )

    # 6. 평가 진행
    eval_qa_list = [qa for qa in full_qa_list if not qa.positive_doc_ids.isdisjoint(target_doc_ids)]
    print(f"\n🔎 평가 시작 (총 {len(eval_qa_list)}개 질문)")
    
    metrics = {f"recall@{k}": 0.0 for k in TOP_K_LIST}
    metrics.update({f"mrr@{k}": 0.0 for k in TOP_K_LIST})

    for i, ex in enumerate(tqdm(eval_qa_list, desc="Evaluating")):
        try:
            initial_nodes = hybrid_retriever.retrieve(ex.question)
            
            reranked_nodes = reranker.postprocess_nodes(
                initial_nodes, 
                query_str=ex.question
            )
            final_nodes = reranked_nodes
            
        except Exception as e:
            # print(f"Error: {e}")
            final_nodes = []
        
        retrieved_doc_ids = [node.metadata.get("doc_id", "") for node in final_nodes]
        gt_set = ex.positive_doc_ids

        if i < 3:
            tqdm.write(f"\n[Q] {ex.question}")
            tqdm.write(f"   Target: {list(gt_set)}")
            tqdm.write(f"   Pred:   {retrieved_doc_ids[:5]}")
            hit = any(d in gt_set for d in retrieved_doc_ids[:5])
            tqdm.write(f"   -> {'✅ HIT' if hit else '❌ MISS'}")

        for k in TOP_K_LIST:
            current_top_k = retrieved_doc_ids[:min(k, len(retrieved_doc_ids))]
            if any(did in gt_set for did in current_top_k):
                metrics[f"recall@{k}"] += 1.0
            for rank, did in enumerate(current_top_k, start=1):
                if did in gt_set:
                    metrics[f"mrr@{k}"] += 1.0 / rank
                    break

    # 최종 결과
    count = len(eval_qa_list)
    print("\n" + "="*50)
    print(f"📈 최종 Fast Graph 평가 결과 (Samples: {count})")
    print("="*50)
    
    if count > 0:
        for k in TOP_K_LIST:
            recall = metrics[f'recall@{k}'] / count
            mrr = metrics[f'mrr@{k}'] / count
            print(f"Recall@{k:<2}   | {recall:.4f}")
            print(f"MRR@{k:<2}      | {mrr:.4f}")

if __name__ == "__main__":
    evaluate_system()