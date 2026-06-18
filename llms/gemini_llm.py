import google.generativeai as genai
from utils.prompts import RAG_PROMPT


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
    ) -> None:
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(
        self,
        context: str,
        question: str,
    ) -> str:
        prompt = RAG_PROMPT.format(
            context=context,
            question=question,
        )
        response = self.model.generate_content(prompt)

        text = response.text
        return text
