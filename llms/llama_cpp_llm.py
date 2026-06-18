from llama_cpp import Llama
from utils.prompts import RAG_PROMPT
from typing import cast
from llama_cpp.llama_types import CreateCompletionResponse


class LlamaCppLLM:
    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            verbose=False,
        )

    def generate(
        self,
        context: str,
        question: str,
    ) -> str:
        prompt = RAG_PROMPT.format(
            context=context,
            question=question,
        )

        output: CreateCompletionResponse = cast(
            CreateCompletionResponse,
            self.llm(
                prompt,
                max_tokens=512,
                temperature=0.1,
                stop=["Question:"],
            ),
        )

        text: str = output["choices"][0]["text"]
        return text.strip()
