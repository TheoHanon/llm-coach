import os
import pathlib
from dotenv import load_dotenv
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain.schema import Document


ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "corpus"
INDEX_DIR = ROOT / "data" / "index_faiss"

load_dotenv(dotenv_path=(ROOT / ".env").expanduser())
INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)


def load_docs() -> list[Document]:
    return DirectoryLoader(
        str(CORPUS_DIR), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    ).load()


def enrich_metadata(docs: List[Document]) -> List[Document]:
    for d in docs:
        src = d.metadata.get("source", "")
        if src:
            p = pathlib.Path(src)
            d.metadata.setdefault("filename", p.name)
            d.metadata.setdefault("title", p.stem.replace("_", " ").title())
            try:
                d.metadata.setdefault(
                    "category", pathlib.Path(src).relative_to(CORPUS_DIR).parts[0]
                )
            except Exception:
                pass
    return docs


def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def build_index(chunks: List[Document]):
    emb = MistralAIEmbeddings(
        model="mistral-embed", api_key=os.environ["MISTRAL_API_KEY"]
    )
    vs = FAISS.from_documents(chunks, emb)
    vs.save_local(str(INDEX_DIR))


if __name__ == "__main__":
    docs = enrich_metadata(load_docs())
    chunks = split_docs(docs)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks")
    build_index(chunks)
    print(f"Saved FAISS index to {INDEX_DIR}")
