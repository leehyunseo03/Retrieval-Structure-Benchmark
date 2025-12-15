import os
import json
import ujson
import fitz  # PyMuPDF
import time
import random
import shutil
import hashlib
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any, Set, Optional
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
import asyncio
from typing import List, Dict, Any, Generator

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
from llama_index.core.llms import CustomLLM, LLMMetadata, CompletionResponse, ChatResponse
from llama_index.core.base.llms.types import ChatMessage
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI as OpenAIClient # Batch 업로드용
from google import genai

# =========================================================
# [설정 영역] API KEY 및 경로
# =========================================================

load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
BASE_DIR = os.environ.get('BASE_DIR') 
PDF_DIR = os.path.join(BASE_DIR, "pdfs_dev")
DEV_QA_PATH = os.path.join(BASE_DIR, "multimodalqa", "MMQA_dev.jsonl")

# --- [Batch 설정] ---
# 실행 모드 선택: "RECORD" (요청 파일 생성) or "REPLAY" (결과로 평가)
# 최초 실행 시 "RECORD" -> Batch 업로드/완료 -> "REPLAY"로 변경 후 실행
#EXECUTION_MODE = "RECORD" 
EXECUTION_MODE = "REPLAY"

BATCH_INPUT_FILE = "batch_input.jsonl"   # 생성될 요청 파일
BATCH_OUTPUT_FILE = "batch_output.jsonl" # 다운로드 받은 결과 파일
MODEL_NAME = "gpt-4o-mini"
# ------------------------

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
client = genai.Client(api_key=GOOGLE_API_KEY)
vision_model_id = 'gemini-2.0-flash'

TOP_K_LIST = [1, 3, 5, 10]
LIMIT_PDFS = 10

# =========================================================
# [Part 1] Batch API 지원용
# =========================================================
class BatchMockLLM(CustomLLM):
    """
    OpenAI Batch API용 래퍼. 
    chat, complete, async 호출 등 모든 경로를 가로채서 파일에 기록합니다.
    """
    mode: str = "RECORD"
    requests_buffer: List[Dict] = []
    responses_cache: Dict[str, str] = {}
    model_name: str = MODEL_NAME

    def __init__(self, mode: str, output_file: str = None):
        super().__init__()
        self.mode = mode
        if mode == "REPLAY" and output_file and os.path.exists(output_file):
            print(f"📂 Batch 결과 파일 로드 중: {output_file}")
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        cid = data.get("custom_id")
                        content = data["response"]["body"]["choices"][0]["message"]["content"]
                        self.responses_cache[cid] = content
                    except Exception as e:
                        pass
            print(f"✅ {len(self.responses_cache)}개의 응답 로드 완료.")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model_name)

    def _get_hash(self, messages: List[ChatMessage]) -> str:
        """메시지 내용으로 고유 ID 생성"""
        content_str = "".join([f"{m.role}:{m.content}" for m in messages])
        return hashlib.md5(content_str.encode("utf-8")).hexdigest()

    # --- [핵심 로직] 파일 기록 및 가로채기 ---
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        custom_id = self._get_hash(messages)

        if self.mode == "RECORD":
            req = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_name,
                    "messages": [{"role": m.role.value, "content": m.content} for m in messages],
                    "temperature": 0
                }
            }
            # 파일에 즉시 쓰기
            with open(BATCH_INPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
            
            # 더미 응답 (JSON 파싱 에러 방지용)
            return ChatResponse(message=ChatMessage(role="assistant", content="[]"))

        elif self.mode == "REPLAY":
            content = self.responses_cache.get(custom_id, "[]")
            return ChatResponse(message=ChatMessage(role="assistant", content=content))

    # --- [리다이렉트] 다른 메서드 호출 시에도 무조건 chat 로직을 태움 ---
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        # complete 요청이 오면 user message로 감싸서 chat으로 보냄
        msg = ChatMessage(role="user", content=prompt)
        chat_response = self.chat([msg], **kwargs)
        return CompletionResponse(text=chat_response.message.content)

    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        # 비동기 요청도 동기 chat 메서드로 연결 (파일 쓰기는 blocking이어도 무방)
        return self.chat(messages, **kwargs)

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    # --- [인터페이스 준수용] ---
    def stream_chat(self, messages: List[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        yield self.chat(messages, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        yield self.complete(prompt, **kwargs)

# =========================================================
# [Part 2] 데이터 로더 (기존과 동일)
# =========================================================

@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]
    positive_doc_ids: Set[str]

def load_mmqa_data(jsonl_path: str) -> List[QAExample]:
    qa_list = []
    if not os.path.exists(jsonl_path): return []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            ex = ujson.loads(line)
            positive_doc_ids = set()
            if "supporting_context" in ex:
                for ctx in ex["supporting_context"]:
                    if ctx.get("doc_id"): positive_doc_ids.add(str(ctx.get("doc_id")))
            
            if positive_doc_ids:
                qa_list.append(QAExample(ex.get("qid"), ex["question"], 
                                       [a["answer"] if isinstance(a, dict) else a for a in ex.get("answers", [])], 
                                       positive_doc_ids))
    return qa_list

class SmartPDFLoader:
    def __init__(self, pdf_dir: str):
        self.pdf_dir = pdf_dir
    
    def load_specific_documents(self, target_doc_ids: Set[str]) -> List[Document]:
        all_docs = []
        available_files = {f.split('.')[0]: f for f in os.listdir(self.pdf_dir) if f.lower().endswith(".pdf")}
        found_files = [available_files[tid] for tid in target_doc_ids if tid in available_files]
        
        for filename in tqdm(found_files, desc="📂 문서 처리 중", unit="file"):
            filepath = os.path.join(self.pdf_dir, filename)
            doc_id = filename.split('.')[0]
            try:
                text_docs = SimpleDirectoryReader(input_files=[filepath]).load_data()
                for d in text_docs: d.metadata["doc_id"] = doc_id
                all_docs.extend(text_docs)
            except: pass
        return all_docs

# =========================================================
# [Part 3] 실행 및 평가
# =========================================================

def upload_batch_file():
    """RECORD 모드로 생성된 파일을 OpenAI에 업로드합니다."""
    if not os.path.exists(BATCH_INPUT_FILE):
        print(f"❌ {BATCH_INPUT_FILE} 파일이 없습니다.")
        return
    
    c = OpenAIClient(api_key=OPENAI_API_KEY)
    print("\n☁️ [OpenAI] Batch 파일 업로드 중...")
    batch_file = c.files.create(file=open(BATCH_INPUT_FILE, "rb"), purpose="batch")
    
    print("🚀 [OpenAI] Batch 작업 생성 중...")
    batch_job = c.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"✅ Batch 작업 시작됨! ID: {batch_job.id}")
    print("⏳ 작업이 완료(completed)되면 결과 파일(output_file_id)을 다운로드하여")
    print(f"   '{BATCH_OUTPUT_FILE}' 이름으로 저장한 뒤 EXECUTION_MODE='REPLAY'로 다시 실행하세요.")

def evaluate_system():
    # 1. 이전 파일 정리 (RECORD 모드일 때만)
    if EXECUTION_MODE == "RECORD" and os.path.exists(BATCH_INPUT_FILE):
        os.remove(BATCH_INPUT_FILE)

    # [핵심 변경] Settings.llm 교체
    Settings.llm = BatchMockLLM(mode=EXECUTION_MODE, output_file=BATCH_OUTPUT_FILE)

    print("\n" + "="*40)
    print(f"🚀 MMQA 시스템 시작 | 모드: {EXECUTION_MODE}")
    print("="*40)

    # --- 데이터 로드 부분 ---
    full_qa_list = load_mmqa_data(DEV_QA_PATH)
    if not full_qa_list: return
    
    needed_doc_ids = set()
    for qa in full_qa_list: needed_doc_ids.update(qa.positive_doc_ids)
    available_pdfs = set(f.split('.')[0] for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))
    valid_doc_ids = list(needed_doc_ids.intersection(available_pdfs))
    
    if not valid_doc_ids:
        print("❌ 매칭 PDF 없음")
        return

    random.seed(42) # 일관성을 위해 시드 고정
    random.shuffle(valid_doc_ids)
    target_doc_ids = set(valid_doc_ids[:LIMIT_PDFS])
    
    loader = SmartPDFLoader(PDF_DIR)
    docs = loader.load_specific_documents(target_doc_ids)
    
    # --- 인덱싱 (여기서 LLM 호출 발생) ---
    print(f"🏗️  Knowledge Graph 처리 중... (Mode: {EXECUTION_MODE})")
    
    # RECORD 모드에서는 여기서 파일만 쓰고, 실제 인덱스는 텅 빈 상태가 됨
    try:
        index = PropertyGraphIndex.from_documents(
            docs,
            embed_model=Settings.embed_model,
            llm=Settings.llm,
            kg_extractors=[SimpleLLMPathExtractor(llm=Settings.llm, max_paths_per_chunk=5)],
            show_progress=True 
        )
    except Exception as e:
        # RECORD 모드일 때 더미 응답 때문에 파싱 에러가 날 수 있으나 무시해도 됨
        if EXECUTION_MODE == "RECORD": pass
        else: print(f"⚠️ Warning: {e}")

    # --- RECORD 모드 종료 처리 ---
    if EXECUTION_MODE == "RECORD":
        print(f"\n✅ [RECORD 완료] '{BATCH_INPUT_FILE}' 생성됨.")
        print("   이제 OpenAI에 업로드합니다...")
        upload_batch_file()
        return

    # --- REPLAY 모드 평가 수행 ---
    retriever = index.as_retriever(include_text=True, similarity_top_k=max(TOP_K_LIST))
    filtered_qa_list = [qa for qa in full_qa_list if not qa.positive_doc_ids.isdisjoint(target_doc_ids)]
    
    print(f"🔎 평가 시작 ({len(filtered_qa_list)} questions)")
    metrics = {f"recall@{k}": 0.0 for k in TOP_K_LIST}
    metrics.update({f"mrr@{k}": 0.0 for k in TOP_K_LIST})
    
    for i, ex in enumerate(tqdm(filtered_qa_list, desc="Evaluaton")):
        try:
            nodes = retriever.retrieve(ex.question)
        except:
            nodes = []

        retrieved_details = []
        retrieved_doc_ids = []  # ID 저장용 리스트

        for node in nodes:
            r_doc_id = node.metadata.get("doc_id", "")
            retrieved_doc_ids.append(r_doc_id) # [버그수정] ID 리스트에 추가
            retrieved_details.append(f"[{r_doc_id}] {node.text[:50]}...")

        gt_set = ex.positive_doc_ids
        
        # [로그 출력]
        tqdm.write(f"\n📌 Q: {ex.question}")
        tqdm.write(f"✅ GT: {list(gt_set)}")
        # [버그수정] retrieved_doc_ids 사용
        hit_check = any(d in gt_set for d in retrieved_doc_ids[:5]) 
        tqdm.write(f"🎯 Hit: {hit_check}")

        for k in TOP_K_LIST:
            current_top_k = retrieved_doc_ids[:k]
            if any(did in gt_set for did in current_top_k):
                metrics[f"recall@{k}"] += 1.0
            for rank, did in enumerate(current_top_k, 1):
                if did in gt_set:
                    metrics[f"mrr@{k}"] += 1.0 / rank
                    break

    # 결과 출력
    count = len(filtered_qa_list)
    print("\n📊 결과")
    if count > 0:
        for k in TOP_K_LIST:
            print(f"K={k:<2} | Recall: {metrics[f'recall@{k}']/count:.4f} | MRR: {metrics[f'mrr@{k}']/count:.4f}")

if __name__ == "__main__":
    evaluate_system()