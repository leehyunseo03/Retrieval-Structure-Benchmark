import os
import sys
import json
import random
import argparse
from typing import List
from tqdm import tqdm
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex, Document
from llama_index.core.indices.property_graph import ImplicitPathExtractor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Settings, STORAGE_DIR, QAExample, LIMIT

BASE_DIR = os.environ.get('QASPER_DIR')
PDF_DIR = os.path.join(BASE_DIR, "qasper_pdfs")
JSON_PATH = os.path.join(BASE_DIR, "qasper", "qasper-dev-v0.3.json")
SAVE_DIR = os.path.join(STORAGE_DIR, "qasper")

def load_qa_data() -> List[QAExample]:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
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

def ingest(limit: int = 0):
    print(f"🚀 [QASPER] Fast Ingestion Start...")
    
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

    print(f"🏗️  Building Property Graph (Chunks: {len(docs)})...")
    
    # [핵심 변경] ImplicitPathExtractor 사용 (LLM 호출 X -> 속도 빠름)
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