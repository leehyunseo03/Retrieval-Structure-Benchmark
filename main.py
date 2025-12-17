#main.py
import argparse
import os
import sys
from tqdm import tqdm
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

# 모듈 동적 임포트를 위해 각 파일을 임포트
import FinanceBench.FinanceBench as FinanceBench
import M3DocVQA.M3DocVQA as M3DocVQA
import QASPER.QASPER as QASPER
from config import Settings, STORAGE_DIR

TOP_K_LIST = [1, 3, 5, 10]

# 데이터셋 이름 문자열과 모듈 매핑
MODULE_MAP = {
    "financebench": FinanceBench,
    "m3docvqa": M3DocVQA,
    "qasper": QASPER
}

def evaluate(dataset_name: str):
    print(f"\n🚀 [Advanced Evaluation] Target: {dataset_name.upper()}")
    
    if dataset_name not in MODULE_MAP:
        print(f"❌ Invalid dataset name. Choose from: {list(MODULE_MAP.keys())}")
        return
    
    target_module = MODULE_MAP[dataset_name]
    persist_dir = target_module.SAVE_DIR

    # 1. 인덱스 로드
    if not os.path.exists(persist_dir):
        print(f"❌ No saved index found at: {persist_dir}")
        print(f"   Please run 'python {dataset_name}/{dataset_name}.py --limit <N>' first.")
        return

    print("📂 Loading Index from Disk...")
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)
    
    # HyDE (Query Expansion)
    print("🧠 Setting up HyDE Query Expansion...")
    hyde = HyDEQueryTransform(include_original=True, llm=Settings.llm)

    # Hybrid Search 구성 (Vector + BM25)
    print("⚙️  Configuring Hybrid Retriever (Vector + BM25)...")
    
    # (1) Vector Retriever (Graph)
    vector_retriever = index.as_retriever(
        include_text=True, 
        similarity_top_k=20 # Fusion을 위해 넉넉히 가져옴
    )
    
    # (2) BM25 Retriever (Keyword)
    # 인덱스 내의 모든 노드를 가져와서 BM25 인덱스를 즉석에서 생성
    all_nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=all_nodes,
        similarity_top_k=20,
        language="english"
    )
    
    # (3) Fusion Retriever
    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        num_queries=1,
        mode=FUSION_MODES.RECIPROCAL_RANK,
        use_async=True,
        similarity_top_k=20
    )

    # 3. Reranker 설정
    print("🎯 Loading Re-ranker Model...")
    #reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_model = "BAAI/bge-reranker-base"
    reranker = SentenceTransformerRerank(
        model=reranker_model,
        top_n=max(TOP_K_LIST)
    )

    # 4. 저장된 문서 ID 확인 및 QA 필터링
    stored_doc_ids = set()
    for doc in all_nodes:
        if "doc_id" in doc.metadata:
            stored_doc_ids.add(doc.metadata["doc_id"])
            
    print("📖 Loading QA Data...")
    full_qa_list = target_module.load_qa_data()
    
    filtered_qa_list = []
    for qa in full_qa_list:
        if not qa.positive_doc_ids.isdisjoint(stored_doc_ids):
            filtered_qa_list.append(qa)
            
    print(f"✂️  Evaluating {len(filtered_qa_list)} valid questions (out of {len(full_qa_list)})")
    if not filtered_qa_list:
        return

    # 5. 평가 루프
    metrics = {f"recall@{k}": 0.0 for k in TOP_K_LIST}
    metrics.update({f"mrr@{k}": 0.0 for k in TOP_K_LIST})

    for i, ex in enumerate(tqdm(filtered_qa_list, desc="Retrieving & Reranking")):
        try:
            query_bundle = hyde(ex.question)

            # Hybrid Search
            initial_nodes = hybrid_retriever.retrieve(query_bundle)
            
            # Re-ranking
            final_nodes = reranker.postprocess_nodes(
                initial_nodes, 
                query_str=ex.question
            )
        except Exception:
            final_nodes = []
        
        retrieved_ids = [node.metadata.get("doc_id", "") for node in final_nodes]
        gt_set = ex.positive_doc_ids

        # 디버깅 (첫 샘플)
        if i == 0:
            tqdm.write(f"\n[Sample Q] {ex.question}")
            tqdm.write(f"   GT: {list(gt_set)}")
            tqdm.write(f"   Pred: {retrieved_ids[:5]}")
            hit = any(d in gt_set for d in retrieved_ids[:5])
            tqdm.write(f"   Result: {'✅ HIT' if hit else '❌ MISS'}")

        for k in TOP_K_LIST:
            current_top_k = retrieved_ids[:k]
            # Recall
            if any(did in gt_set for did in current_top_k):
                metrics[f"recall@{k}"] += 1.0
            # MRR
            for rank, did in enumerate(current_top_k, start=1):
                if did in gt_set:
                    metrics[f"mrr@{k}"] += 1.0 / rank
                    break
    
    # 결과 출력
    count = len(filtered_qa_list)
    print("\n" + "="*50)
    print(f"📈 Advanced Evaluation Results: {dataset_name} (n={count})")
    print("="*50)
    if count > 0:
        for k in TOP_K_LIST:
            rec = metrics[f'recall@{k}'] / count
            mrr = metrics[f'mrr@{k}'] / count
            print(f"Recall@{k:<2} | {rec:.4f}")
            print(f"MRR@{k:<2}    | {mrr:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=MODULE_MAP.keys(), help="Dataset to evaluate")
    args = parser.parse_args()
    
    evaluate(args.dataset)