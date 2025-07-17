from __future__ import annotations

import os
import pathlib
from typing import List
from tqdm import tqdm

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel

from dotenv import load_dotenv
load_dotenv()


def _choose_llm() -> BaseLanguageModel:
    """Choose LLM backend based on env var LLM_SOURCE (anthropic|ollama)."""
    llm_source = os.getenv("LLM_SOURCE", "ollama").lower()
    if llm_source == "anthropic":
        return ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
    elif llm_source == "ollama":
        return OllamaLLM(model="gemma3:4b", temperature=0)
    else:
        raise ValueError(f"Invalid LLM source: {llm_source}")


def _choose_embedding() -> "OpenAIEmbeddings | OllamaEmbeddings":
    """Return an embedding model instance."""
    emb_source = os.getenv("EMBEDDING_SOURCE", "openai").lower()
    if emb_source == "ollama":
        return OllamaEmbeddings(model="nomic-embed-text")
    elif emb_source == "openai":
        return OpenAIEmbeddings()
    else:
        raise ValueError(f"Invalid embedding source: {emb_source}")


class CodebaseTourGuide:
    """Agent that indexes a code repository and answers location questions."""

    def __init__(
        self,
        repo_path: str,
        index_path: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.repo_path = pathlib.Path(repo_path).resolve()
        if not self.repo_path.is_dir():
            raise ValueError(f"{self.repo_path} is not a directory")

        self.index_path = (
            pathlib.Path(index_path) if index_path else self.repo_path / ".code_index"
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separators=["\n"]
        )

        self.embedding_model = _choose_embedding()
        self.vectorstore: FAISS | None = None

        # Load existing index if present, otherwise build
        if (self.index_path / "index.faiss").exists():
            print(f"Loading existing index from {self.index_path}...")
            self.vectorstore = FAISS.load_local(str(self.index_path), self.embedding_model)
            print("Index loaded successfully!")
        else:
            print("No existing index found. Building new index...")
            self.build_index()

    # ---------------------------------------------------------------------
    # Index building
    # ---------------------------------------------------------------------
    def _iter_source_files(self) -> List[pathlib.Path]:
        """Return list of relevant source files (code, md) in repo."""
        exts = {".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml"}
        files: List[pathlib.Path] = []
        for p in tqdm(self.repo_path.rglob("*"), desc="Discovering files"):
            if p.suffix in exts and p.is_file():
                files.append(p)
        return files

    def build_index(self) -> None:
        """Create a FAISS vector store from repo contents."""
        docs: List[Document] = []
        source_files = self._iter_source_files()
        
        print(f"Found {len(source_files)} source files to process...")
        
        for file_path in tqdm(source_files, desc="Processing files"):
            try:
                text = file_path.read_text(errors="ignore")
            except Exception:
                continue  # skip binary/unreadable files
            # Split text into chunks with metadata path + line numbers
            splits = self.text_splitter.split_text(text)
            for i, chunk in enumerate(splits):
                metadata = {
                    "source": str(file_path.relative_to(self.repo_path)),
                    "chunk": i,
                }
                docs.append(Document(page_content=chunk, metadata=metadata))

        print(f"Created {len(docs)} document chunks. Building vector store...")
        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.vectorstore.save_local(str(self.index_path))
        print(f"Index saved to {self.index_path}")

    # ---------------------------------------------------------------------
    # Query answer
    # ---------------------------------------------------------------------
    def query(self, question: str, k: int = 6) -> str:
        """Return an LLM-generated answer citing the most relevant snippets."""
        if self.vectorstore is None:
            raise RuntimeError("Vector store not built")

        top_docs = self.vectorstore.similarity_search(question, k=k)
        context_blocks: List[str] = []
        for doc in top_docs:
            source = doc.metadata.get("source", "<unknown>")
            context_blocks.append(f"# File: {source}\n{doc.page_content}")
        context = "\n\n".join(context_blocks)

        prompt_tmpl = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are a senior software engineer acting as a Codebase Tour Guide.\nYou will answer developer questions about a repository using the provided context.\nIf the answer is not in the context, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer as markdown with citations to the file names.""",
        )

        llm = _choose_llm()
        chain = prompt_tmpl | llm
        return chain.invoke({"question": question, "context": context})

    # Convenience entry point
    def __call__(self, question: str, k: int = 6) -> str:  # noqa: D401
        """Alias for `query`."""
        return self.query(question, k=k)


def main():
    """Main entry point for the Codebase Tour Guide agent."""
    print("=== Codebase Tour Guide ===")
    print("This agent indexes a codebase and answers questions like\n"
          "\"Where is the API authentication handled?\" by retrieving the most\n"
          "relevant source files / snippets and optionally passing them to an LLM\n"
          "for synthesis.\n")
    
    # Initialize agent
    repo_path = input("Enter the path to the codebase: ").strip()

    agent = CodebaseTourGuide(repo_path)

    while True:
        question = input("Enter your question: ").strip()
        if not question:
            break
        answer = agent.query(question)
        print("\nAnswer:\n", answer)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()