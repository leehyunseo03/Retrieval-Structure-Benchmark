#M3DocVQA/M3DocVQA.py
import os
import sys
import ujson
import random
import argparse
from typing import List
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Document
from llama_index.core.indices.property_graph import ImplicitPathExtractor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings, STORAGE_DIR, QAExample, LIMIT

BASE_DIR = os.environ.get('M3DocVQA_DIR')
PDF_DIR = os.path.join(BASE_DIR, "pdfs_dev")
JSONL_PATH = os.path.join(BASE_DIR, "multimodalqa", "MMQA_dev.jsonl")
SAVE_DIR = os.path.join(STORAGE_DIR, "m3docvqa")

def load_qa_data() -> List[QAExample]:
    qa_list = []
    if not os.path.exists(JSONL_PATH):
        print(f"❌ File not found: {JSONL_PATH}")
        return []

    with open(JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = ujson.loads(line)
            
            pids = set()
            if "supporting_context" in ex:
                for ctx in ex["supporting_context"]:
                    if ctx.get("doc_id"): pids.add(str(ctx.get("doc_id")))
            
            if pids:
                qa_list.append(QAExample(
                    qid=str(ex.get("qid") or ex.get("id")),
                    question=ex["question"],
                    answers=[a["answer"] if isinstance(a, dict) else a for a in ex.get("answers", [])],
                    positive_doc_ids=pids
                ))
    return qa_list

def ingest(limit: int = 0):
    print(f"🚀 [M3DocVQA] Fast Ingestion Start...")
    
    qa_list = load_qa_data()
    needed_ids = set()
    for qa in qa_list:
        needed_ids.update(qa.positive_doc_ids)

    available_files = {f.split('.')[0]: f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')}
    valid_ids = list(needed_ids.intersection(available_files.keys()))

    if limit > 0:
        target_ids = random.sample(valid_ids, min(limit, len(valid_ids)))
        print(f"✂️  Limiting to {len(target_ids)} documents.")
    else:
        target_ids = valid_ids

    docs = []
    print(f"📂 Loading {len(target_ids)} PDFs...")
    for doc_id in tqdm(target_ids, desc="Loading PDFs"):
        filepath = os.path.join(PDF_DIR, available_files[doc_id])
        try:
            loaded = SimpleDirectoryReader(input_files=[filepath]).load_data()
            
            # [핵심 수정 부분]
            # 기존 객체(d)를 수정하지 않고, 내용을 가져와서 '새로운 Document'를 만듭니다.
            for d in loaded:
                original_text = d.get_content()
                new_text = f"[Paper ID: {doc_id}]\n{original_text}"
                
                # 메타데이터 복사 및 추가
                new_metadata = d.metadata.copy()
                new_metadata["doc_id"] = doc_id
                new_metadata["file_name"] = available_files[doc_id]
                
                # 새 객체 생성하여 추가
                new_doc = Document(text=new_text, metadata=new_metadata)
                docs.append(new_doc)
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    print(f"🏗️  Building Fast Property Graph (Chunks: {len(docs)})...")
    
    # [핵심 변경] ImplicitPathExtractor 사용
    index = PropertyGraphIndex.from_documents(
        docs,
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        kg_extractors=[ImplicitPathExtractor()], 
        show_progress=True
    )
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=SAVE_DIR)
    print(f"💾 Saved index to: {SAVE_DIR}")

if __name__ == "__main__":
    ingest(limit=LIMIT)