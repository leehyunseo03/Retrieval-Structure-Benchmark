import os
from dataclasses import dataclass
from typing import List, Set
from dotenv import load_dotenv

from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# [1] 공통 모델 설정
# 임베딩 모델이 바뀌면 저장된 인덱스를 못 쓰므로 한곳에서 관리합니다.
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# [2] 저장소 경로
STORAGE_DIR = "./saved_indices"
LIMIT = 30

# [3] 데이터 클래스
@dataclass
class QAExample:
    qid: str
    question: str
    answers: List[str]
    positive_doc_ids: Set[str]