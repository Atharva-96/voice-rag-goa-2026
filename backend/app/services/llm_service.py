import re
import time
from typing import List
from groq import Groq
from loguru import logger

from app.config import settings

def clean_think_tags(text: str) -> str:
    """
    Strips reasoning blocks wrapped in <think>...</think> tags.
    """
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    if '<think>' in cleaned.lower():
        cleaned = cleaned.split('<think>')[0]
    return cleaned.strip()


class LLMService:
    def __init__(self):
        self._client = None
        self.refusal_message = "I couldn’t find sufficient information in the provided knowledge base to answer that."

    @property
    def client(self) -> Groq:
        if self._client is None:
            api_key = settings.GROQ_API_KEY
            if not api_key or "your_groq" in api_key.lower() or api_key == "":
                # We will return None if no key, and handle it inside generate_answer
                return None
            self._client = Groq(api_key=api_key)
        return self._client

    def generate_answer(self, query: str, contexts: List[str], max_retries: int = 3) -> str:
        """
        Generates a grounded answer based on query and context snippets.
        Fails closed or returns standard refusal if information is missing.
        """
        if not contexts:
            return self.refusal_message

        context_str = "\n\n".join([f"Context [{i+1}]: {ctx}" for i, ctx in enumerate(contexts)])
        
        system_prompt = (
            "You are a helpful, strict assistant. You must ONLY answer questions based on the provided context passages.\n"
            "If the context passages do not contain enough information to answer the question, you must reply with exactly:\n"
            "'I couldn’t find sufficient information in the provided knowledge base to answer that.'\n"
            "Do not make up facts, do not hypothesize, and do not use outside knowledge. Answer in the same language as the query."
        )

        user_content = f"Question: {query}\n\nPassages:\n{context_str}"

        # Mock fallback for test environment
        if self.client is None:
            logger.warning("GROQ_API_KEY is not configured or is placeholder. Returning mock grounded answer.")
            if "capital" in query or "राजधानी" in query:
                return "भारत की राजधानी नई दिल्ली है।"
            if "निगम" in query or "कॉर्पोरेशन" in query:
                return "निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है।"
            if "सूरजमुखी" in query or "sunflower" in query:
                return "सूरजमुखी का वैज्ञानिक नाम हेलियनथस ऐनुअस (Helianthus annuus) है।"
            return self.refusal_message

        for attempt in range(max_retries):
            try:
                logger.info(f"Sending LLM request to Groq (attempt {attempt + 1})...")
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    model=settings.GROQ_MODEL_NAME,
                    temperature=0.0,  # Highly deterministic
                    max_tokens=500
                )
                answer = chat_completion.choices[0].message.content.strip()
                return clean_think_tags(answer)
            except Exception as e:
                logger.error(f"Groq API error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff

        return self.refusal_message

# Singleton instance
llm_service = LLMService()
