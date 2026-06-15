# RAG_Core/models/llm_model.py - RAW OLLAMA STREAMING

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import settings
import traceback
from typing import Iterator, Optional, AsyncIterator
import logging
import httpx
import json

logger = logging.getLogger(__name__)


class LLMModel:
    def __init__(self):
        logger.info(f"[LLMModel] Using model={settings.LLM_MODEL} base_url={getattr(settings, 'OLLAMA_URL', None)}")

        # Initialize Ollama LLM for non-streaming
        self.llm = OllamaLLM(
            model=settings.LLM_MODEL,
            base_url=getattr(settings, "OLLAMA_URL", "http://ollama:11434"),
            temperature=0.1,
        )
        self.output_parser = StrOutputParser()

        # Ollama API endpoint
        self.ollama_url = getattr(settings, "OLLAMA_URL", "http://ollama:11434")
        self.model_name = settings.LLM_MODEL

    def invoke(self, prompt: str, **kwargs) -> str:
        """Non-streaming invoke"""
        try:
            resp = self.llm.invoke(prompt, **kwargs)
            return self.output_parser.parse(resp)
        except Exception as e:
            traceback.print_exc()
            return f"Lỗi xử lý: {str(e)}"

    def stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Sync streaming using raw Ollama API"""
        try:
            logger.info(f"Starting sync stream with raw Ollama API...")

            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1
                }
            }

            with httpx.stream("POST", url, json=payload, timeout=60.0) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk_text = data.get("response", "")

                            if chunk_text:
                                yield chunk_text

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            traceback.print_exc()
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        FIXED: Async streaming using raw Ollama API
        """
        try:
            logger.info(f"🚀 Starting async stream with raw Ollama API...")
            logger.info(f"📝 Prompt length: {len(prompt)}")

            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1
                }
            }

            chunk_count = 0
            total_text = ""

            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                chunk_text = data.get("response", "")

                                if chunk_text:
                                    chunk_count += 1
                                    total_text += chunk_text
                                    logger.debug(f"✅ Chunk #{chunk_count}: '{chunk_text[:30]}...'")
                                    yield chunk_text

                                # Check if done
                                if data.get("done", False):
                                    logger.info(f"🏁 Stream completed")
                                    break

                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse JSON line: {e}")
                                continue

            logger.info(f"📊 Total chunks: {chunk_count}")
            logger.info(f"📊 Total text length: {len(total_text)}")

        except Exception as e:
            logger.error(f"❌ Async streaming FAILED: {e}", exc_info=True)
            traceback.print_exc()
            yield f"\n\n[Lỗi streaming: {str(e)}]"

    def create_chain(self, template: str):
        """Create a non-streaming chain"""
        prompt = PromptTemplate.from_template(template)
        return prompt | self.llm | self.output_parser


# Global instance
llm_model = LLMModel()