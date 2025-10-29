import os, json, uuid
from types import SimpleNamespace
from typing import Any, Dict, List

import requests
from google import generativeai as genai
from google.generativeai import types, GenerationConfig

from core.providers.llm.base import LLMProviderBase
from core.utils.util import check_model_key
from config.logger import setup_logging
from google.generativeai.types import GenerateContentResponse
from requests import RequestException

log = setup_logging()
TAG = __name__


def test_proxy(proxy_url: str, test_url: str) -> bool:
    try:
        resp = requests.get(test_url, proxies={"http": proxy_url, "https": proxy_url})
        return 200 <= resp.status_code < 400
    except RequestException:
        return False


def setup_proxy_env(http_proxy: str | None, https_proxy: str | None):
    """
    分别测试 HTTP 和 HTTPS 代理是否可用，并设置环境变量。
    如果 HTTPS 代理不可用但 HTTP 可用，会将 HTTPS_PROXY 也指向 HTTP。
    """
    test_http_url = "http://www.google.com"
    test_https_url = "https://www.google.com"

    ok_http = ok_https = False

    if http_proxy:
        ok_http = test_proxy(http_proxy, test_http_url)
        if ok_http:
            os.environ["HTTP_PROXY"] = http_proxy
            log.bind(tag=TAG).info(f"配置提供的Gemini HTTPS代理连通成功: {http_proxy}")
        else:
            log.bind(tag=TAG).warning(f"配置提供的Gemini HTTP代理不可用: {http_proxy}")

    if https_proxy:
        ok_https = test_proxy(https_proxy, test_https_url)
        if ok_https:
            os.environ["HTTPS_PROXY"] = https_proxy
            log.bind(tag=TAG).info(f"配置提供的Gemini HTTPS代理连通成功: {https_proxy}")
        else:
            log.bind(tag=TAG).warning(
                f"配置提供的Gemini HTTPS代理不可用: {https_proxy}"
            )

    # 如果https_proxy不可用，但http_proxy可用且能走通https，则复用http_proxy作为https_proxy
    if ok_http and not ok_https:
        if test_proxy(http_proxy, test_https_url):
            os.environ["HTTPS_PROXY"] = http_proxy
            ok_https = True
            log.bind(tag=TAG).info(f"复用HTTP代理作为HTTPS代理: {http_proxy}")

    if not ok_http and not ok_https:
        log.bind(tag=TAG).error(
            f"Gemini 代理设置失败: HTTP 和 HTTPS 代理都不可用，请检查配置"
        )
        raise RuntimeError("HTTP 和 HTTPS 代理都不可用，请检查配置")


class LLMProvider(LLMProviderBase):
    def __init__(self, cfg: Dict[str, Any]):
        log.info("Initializing GeminiLLM provider...")
        self.model_name = cfg.get("model_name", "gemini-2.0-flash")
        self.api_key = cfg["api_key"]
        http_proxy = cfg.get("http_proxy")
        https_proxy = cfg.get("https_proxy")

        model_key_msg = check_model_key("LLM", self.api_key)
        if model_key_msg:
            log.error(model_key_msg)

        if http_proxy or https_proxy:
            log.info("Proxy configuration detected for GeminiLLM, testing connectivity...")
            setup_proxy_env(http_proxy, https_proxy)
            log.info(f"Proxy setup completed - HTTP: {http_proxy}, HTTPS: {https_proxy}")

        genai.configure(api_key=self.api_key)
        log.info(f"GeminiLLM API key configured. Model name: {self.model_name}")

        self.timeout = cfg.get("timeout", 120)
        log.info(f"Request timeout set to {self.timeout} seconds.")

        self.model = genai.GenerativeModel(self.model_name)
        log.info(f"GeminiLLM GenerativeModel instance created for model: {self.model_name}")

        self.gen_cfg = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=2048,
        )
        log.info("GeminiLLM generation config initialized.")

    @staticmethod
    def _build_tools(funcs: List[Dict[str, Any]] | None):
        if not funcs:
            return None
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=f["function"]["name"],
                        description=f["function"]["description"],
                        parameters=f["function"]["parameters"],
                    )
                    for f in funcs
                ]
            )
        ]

    # Gemini文档提到，无需维护session-id，直接用dialogue拼接而成
    def response(self, session_id, dialogue, **kwargs):
        log.info(f"GeminiLLM response called. Session ID: {session_id}, Dialogue length: {len(dialogue)}")
        yield from self._generate(dialogue, None)

    def response_with_functions(self, session_id, dialogue, functions=None):
        log.info(f"GeminiLLM response_with_functions called. Session ID: {session_id}, Dialogue length: {len(dialogue)}, Functions: {functions is not None}")
        yield from self._generate(dialogue, self._build_tools(functions))

    def _generate(self, dialogue, tools):
        log.info(f"GeminiLLM _generate called. Dialogue length: {len(dialogue)}, Tools: {tools is not None}")
        role_map = {"assistant": "model", "user": "user"}
        contents: list = []
        for m in dialogue:
            r = m["role"]
            log.info(f"Processing message with role: {r}")

            if r == "assistant" and "tool_calls" in m:
                tc = m["tool_calls"][0]
                #log.info(f"Assistant message contains tool call: {tc['function']['name']}")
                #contents.append(
                #    {
                #        "role": "model",
                #        "parts": [
                #            {
                #                "function_call": {
                #                    "name": tc["function"]["name"],
                #                    "args": json.loads(tc["function"]["arguments"]),
                #                }
                #            }
                #        ],
                #    }
                #)
                continue

            if r == "tool":
                log.info("Processing tool message.")
                #contents.append(
                #    {
                #        "role": "model",
                #        "parts": [{"text": str(m.get("content", ""))}],
                #    }
                # )
                continue

            contents.append(
                {
                    "role": role_map.get(r, "user"),
                    "parts": [{"text": str(m.get("content", ""))}],
                }
            )

        # Log the payload and endpoint
        log.info(f"GeminiLLM request payload: {json.dumps(contents, ensure_ascii=False)[:1000]}")
        # Try to print the endpoint used by the Google API (not directly exposed, but we can infer)
        endpoint_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        log.info(f"GeminiLLM API endpoint: {endpoint_url}")

        log.info("Sending request to GeminiLLM model...")
        try:
            stream: GenerateContentResponse = self.model.generate_content(
                contents=contents,
                generation_config=self.gen_cfg,
                # tools=tools,                 # commented out to avoid issues with function calls
                stream=False,                  # changed for testing
                # timeout=self.timeout,        # seems to be broken
            )
        except Exception as e:
            log.error(f"GeminiLLM API call failed: {e}")
            raise

        try:
            for chunk in stream:
                log.info("Received chunk from GeminiLLM model.")
                cand = chunk.candidates[0]
                for part in cand.content.parts:
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        log.info(f"Function call detected in GeminiLLM response: {fc.name}")
                        yield None, [
                            SimpleNamespace(
                                id=uuid.uuid4().hex,
                                type="function",
                                function=SimpleNamespace(
                                    name=fc.name,
                                    arguments=json.dumps(
                                        dict(fc.args), ensure_ascii=False
                                    ),
                                ),
                            )
                        ]
                        return
                    if getattr(part, "text", None):
                        log.info("Text response detected in GeminiLLM chunk.")
                        yield part.text if tools is None else (part.text, None)

        except Exception as e:
            log.error(f"GeminiLLM stream error: {e}")
            raise

        finally:
            if tools is not None:
                log.info("Function-mode stream ended, sending dummy package.")
                yield None, None

    # 关闭stream，预留后续打断对话功能的功能方法，官方文档推荐打断对话要关闭上一个流，可以有效减少配额计费和资源占用
    @staticmethod
    def _safe_finish_stream(stream: GenerateContentResponse):
        log.info("Finishing GeminiLLM stream safely.")
        if hasattr(stream, "resolve"):
            stream.resolve()
        elif hasattr(stream, "close"):
            stream.close()
        else:
            for _ in stream:
                pass
