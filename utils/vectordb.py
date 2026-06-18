from typing import Any, Sequence, cast
import chromadb
from utils.embeddings import EmbeddingModel


class VectorDB:
    def __init__(self) -> None:
        self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(name="documents")
        self.embedding_model = EmbeddingModel()

    def add_documents(
        self,
        chunks: list[str],
    ) -> None:
        embeddings = self.embedding_model.encode(chunks)

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.add(  # type: ignore[arg-type]
            ids=ids,
            documents=chunks,
            embeddings=cast(list[Sequence[float]], embeddings),
        )

    def search(
        self,
        query: str,
        k: int = 4,
    ) -> str:
        query_embedding = self.embedding_model.encode([query])[0]

        results: Any = self.collection.query(  # type: ignore[arg-type]
            query_embeddings=cast(
                list[Sequence[float]],
                [query_embedding],
            ),
            n_results=k,
        )

        documents = results["documents"][0]

        return "\n\n".join(documents)
