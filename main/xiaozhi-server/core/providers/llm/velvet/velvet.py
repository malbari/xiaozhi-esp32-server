import httpx

from config.logger import setup_logging
from core.utils.util import check_model_key
from core.providers.llm.base import LLMProviderBase

TAG = __name__
logger = setup_logging()


class LLMProvider(LLMProviderBase):
    def __init__(self, config):
        self.api_key = config.get("api_key")
        self.model_name = config.get("model_name")
        self.base_url = config.get("base_url")
        self.skil_llm = config.get("skip_lml")
        logger.debug("Velvet Initialized")

    def response(self, session_id, dialogue, **kwargs):
        logger.debug(f"[Velvet] response called with session_id={session_id}")
        logger.debug(f"[Velvet] Raw dialogue: {dialogue}")
        user_prompt = None
        try:
            # Extract the user prompt (last message with role 'user')
            if not isinstance(dialogue, list):
                logger.debug("[Velvet] Dialogue is not a list, trying eval/json.loads")
                import json
                try:
                    dialogue = json.loads(dialogue)
                except Exception as e:
                    logger.error(f"[Velvet] Unable to convert dialogue to list: {e}")
                    return "Error: invalid dialogue."
            logger.debug(f"[Velvet] Dialogue as list: {dialogue}")
            # Find the last 'user' message
            for msg in reversed(dialogue):
                if msg.get("role") == "user":
                    # Support both 'parts' and 'content' formats
                    if "parts" in msg:
                        parts = msg.get("parts", [])
                        for part in parts:
                            if "text" in part:
                                user_prompt = part["text"]
                                logger.debug(f"[Velvet] User prompt found (parts): {user_prompt}")
                                break
                    elif "content" in msg:
                        user_prompt = msg["content"]
                        logger.debug(f"[Velvet] User prompt found (content): {user_prompt}")
                    if user_prompt:
                        break
            if not user_prompt:
                logger.error("[Velvet] No user prompt found in dialogue.")
                return "Error: user prompt not found."

            # Prepare the request to Velvet LLM
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            payload = {
                "model": self.model_name or "velvet-14b",
                "messages": [{"role": "user", "content": user_prompt}]
            }
            logger.debug(f"[Velvet] LLM call: url={self.base_url}, payload={payload}")
            # If skip_llm is True, return echo message instead of calling LLM
            if self.skil_llm:
                logger.debug(f"[Velvet] skil_llm is True, returning echo message.")
                return f"Hai detto: <<{user_prompt}>>"
            try:
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(self.base_url, headers=headers, json=payload)
                    logger.debug(f"[Velvet] Status code: {response.status_code}")
                    logger.debug(f"[Velvet] Response text: {response.text}")
                    response.raise_for_status()
                    result = response.json()
                    logger.debug(f"[Velvet] Response JSON: {result}")
                    content = result["choices"][0]["message"]["content"].strip()
                    logger.debug(f"[Velvet] Extracted content: {content}")
                    return content
            except Exception as e:
                logger.error(f"[Velvet] Error contacting LLM: {e}")
                return "Error: could not contact Velvet."
        except Exception as e:
            logger.error(f"[Velvet] General error: {e}")
            return "Velvet internal error."
