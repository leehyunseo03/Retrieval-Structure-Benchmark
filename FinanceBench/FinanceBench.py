import os
import sys
import json
import random
import argparse
from typing import List, Set
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Document
from llama_index.core.indices.property_graph import ImplicitPathExtractor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings, STORAGE_DIR, QAExample, LIMIT

# 경로 설정
BASE_DIR = os.environ.get('FinanceBench_DIR')
PDF_DIR = os.path.join(BASE_DIR, "financebench_pdfs")
JSON_PATH = os.path.join(BASE_DIR, "financebench", "financebench_data.json")
SAVE_DIR = os.path.join(STORAGE_DIR, "financebench")

def load_qa_data() -> List[QAExample]:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    qa_list = []
    for filename, content in data.items():
        for qa in content["qas"]:
            qa_list.append(QAExample(
                qid=str(qa["qid"]),
                question=qa["question"],
                answers=[qa["answer"]],
                positive_doc_ids={filename}
            ))
    return qa_list

def ingest(limit: int = 0):
    print(f"🚀 [FinanceBench] Fast Ingestion Start...")
    
    qa_list = load_qa_data()
    needed_ids = set()
    for qa in qa_list:
        needed_ids.update(qa.positive_doc_ids)
    
    available_files = {f.replace('.pdf', ''): f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')}
    valid_ids = list(needed_ids.intersection(available_files.keys()))
    
    if limit > 0:
        target_ids = random.sample(valid_ids, min(limit, len(valid_ids)))
        print(f"✂️  Limiting to {len(target_ids)} documents.")
    else:
        target_ids = valid_ids

    # 문서 로드
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

    # 인덱스 생성
    print(f"🏗️  Building Fast Property Graph (Chunks: {len(docs)})...")
    
    # [핵심 변경] ImplicitPathExtractor 사용
    index = PropertyGraphIndex.from_documents(
        docs,
        embed_model=Settings.embed_model,
        llm=Settings.llm,
        kg_extractors=[ImplicitPathExtractor()], # LLM 호출 없음
        show_progress=True
    )
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=SAVE_DIR)
    print(f"💾 Saved index to: {SAVE_DIR}")

if __name__ == "__main__":
    ingest(limit=LIMIT)