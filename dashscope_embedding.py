import os
from typing import List, Optional

from openai import OpenAI, AsyncOpenAI
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field

#https://help.aliyun.com/zh/model-studio/error-code#apikey-error

class DashScopeV4Embedding(BaseEmbedding):
    """
    LlamaIndex embed_model용 DashScope text-embedding-v4 래퍼 (OpenAI compatible mode).
    - DashScope base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
    - model: text-embedding-v4
    - dimensions 파라미터는 v4에서 지원 (원하면 1024/1536/2048 등으로 지정 가능)
    """

    api_key: str = Field(default="")
    base_url: str = Field(default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    model: str = Field(default="text-embedding-v4")
    dimensions: Optional[int] = Field(default=1024)  # None이면 DashScope 기본값 사용
    encoding_format: str = Field(default="float")

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        model: str = "text-embedding-v4",
        dimensions: Optional[int] = 1024,
        encoding_format: str = "float",
        **kwargs,
    ):
        api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY가 설정되지 않았습니다. env 또는 api_key 인자로 넣어주세요.")

        super().__init__(model_name=model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._aclient = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def _embed(self, inputs):
        params = {
            "model": self.model,
            "input": inputs,
            "encoding_format": self.encoding_format,
        }
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions

        resp = self._client.embeddings.create(**params)
        # resp.data[i].embedding -> List[float]
        return [d.embedding for d in resp.data]

    async def _aembed(self, inputs):
        params = {
            "model": self.model,
            "input": inputs,
            "encoding_format": self.encoding_format,
        }
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions

        resp = await self._aclient.embeddings.create(**params)
        return [d.embedding for d in resp.data]

    # --- BaseEmbedding 필수 구현 ---
    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aembed(query))[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aembed(text))[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return await self._aembed(texts)
