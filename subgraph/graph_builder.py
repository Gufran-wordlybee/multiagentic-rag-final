### Build Index

from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.retrievers import EnsembleRetriever, BM25Retriever
from dotenv import load_dotenv
from subgraph.graph_states import ResearcherState, QueryState
from utils.prompt import GENERATE_QUERIES_SYSTEM_PROMPT
from langchain_core.documents import Document
from typing import Any, Literal, TypedDict, cast

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langgraph.types import Send
from pydantic import BaseModel, Field

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
import logging
from functools import lru_cache
from utils.utils import config

load_dotenv()

logger = logging.getLogger(__name__)

TOP_K = config["retriever"]["top_k"]
TOP_K_COMPRESSION = config["retriever"]["top_k_compression"]
ENSEMBLE_WEIGHTS = config["retriever"]["ensemble_weights"]
COHERE_RERANK_MODEL = config["retriever"]["cohere_rerank_model"]

GROQ_MODEL = config["llm"]["groq_model"]
TEMPERATURE = config["llm"]["temperature"]


# ---------------------------------------------------------------------------
# Per-session retriever construction
# ---------------------------------------------------------------------------
# The retriever used to be built once, at import time, against a single
# hardcoded Chroma collection/PDF. That made it impossible to ever answer
# questions about a freshly-uploaded document without restarting the whole
# process. Now, each session (one uploaded PDF) gets its own Chroma
# collection + persist directory (created by retriever/retriever.py at
# upload time in server.py), and the retriever for that session is built
# lazily here, then cached so repeated turns in the same session don't
# rebuild BM25/vectorstore on every message.
#
# The session's persist_directory/collection_name are threaded through via
# RunnableConfig["configurable"], which is the standard LangGraph mechanism
# for passing per-invocation context into graph nodes.


def _load_documents(vectorstore: Chroma) -> list[Document]:
    """
    Load documents and metadata from the vector store and return them as Langchain Document objects.

    Args:
        vectorstore (Chroma): The vector store instance.

    Returns:
        list[Document]: A list of Document objects containing the content and metadata.
    """
    all_data = vectorstore.get(include=["documents", "metadatas"])
    documents: list[Document] = []

    for content, meta in zip(all_data["documents"], all_data["metadatas"]):
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise ValueError(f"Expected metadata to be a dict, but got {type(meta)}")

        documents.append(Document(page_content=content, metadata=meta))

    return documents


def _build_retrievers(documents: list[Document], vectorstore: Chroma) -> ContextualCompressionRetriever:
    """
    Build and return a compression retriever that includes
    an ensemble retriever and Cohere-based contextual compression.

    Args:
        documents (list[Document]): List of Document objects.
        vectorstore (Chroma): The vector store to use for building retrievers.

    Returns:
        ContextualCompressionRetriever: A compression retriever that can be used to fetch and re-rank documents.
    """
    # Create base retrievers
    retriever_bm25 = BM25Retriever.from_documents(documents, search_kwargs={"k": TOP_K})
    retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K})

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
        weights=ENSEMBLE_WEIGHTS,
    )

    # Set up Cohere re-ranking
    compressor = CohereRerank(top_n=TOP_K_COMPRESSION, model=COHERE_RERANK_MODEL)

    # Build compression retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever,
    )

    return compression_retriever


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once (it's the same model for every session) and reuse it."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=64)
def _get_compression_retriever(
    collection_name: str, persist_directory: str
) -> ContextualCompressionRetriever:
    """Build (and cache) the ensemble + Cohere re-ranking retriever for one session.

    Cached per (collection_name, persist_directory) pair so that multiple
    turns of the same chat session reuse the same retriever instead of
    reloading the vectorstore and rebuilding BM25 on every message.
    """
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embeddings(),
        persist_directory=persist_directory,
    )
    documents = _load_documents(vectorstore)
    if not documents:
        raise ValueError(
            f"No documents found in collection '{collection_name}'. "
            "The session's PDF may not have finished indexing yet."
        )
    return _build_retrievers(documents, vectorstore)


def get_retriever_for_session(config: RunnableConfig) -> ContextualCompressionRetriever:
    """Resolve the compression retriever for the current session from RunnableConfig.

    Args:
        config (RunnableConfig): The LangGraph run config. Expects
            config["configurable"]["collection_name"] and
            config["configurable"]["persist_directory"] to be set by the
            caller (the FastAPI server sets these per chat session).

    Returns:
        ContextualCompressionRetriever: The retriever scoped to this session's document.
    """
    configurable = (config or {}).get("configurable", {})
    collection_name = configurable.get("collection_name")
    persist_directory = configurable.get("persist_directory")

    if not collection_name or not persist_directory:
        raise ValueError(
            "Missing 'collection_name'/'persist_directory' in RunnableConfig.configurable. "
            "Every graph invocation must specify which session's document to query."
        )

    return _get_compression_retriever(collection_name, persist_directory)


def clear_session_retriever_cache(collection_name: str, persist_directory: str) -> None:
    """Evict a session's cached retriever, e.g. after its PDF is deleted/replaced."""
    try:
        _get_compression_retriever.cache_discard((collection_name, persist_directory))  # type: ignore[attr-defined]
    except AttributeError:
        # functools.lru_cache has no per-key discard; clear everything as a fallback.
        _get_compression_retriever.cache_clear()


async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the question (a step in the research plan).

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(BaseModel):
        queries: list[str] = Field(
            description="A list of diverse search queries to help answer the question"
        )

    logger.info("---GENERATE QUERIES---")
    # model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    model = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE)
    # model = ChatGroq(model=GROQ_MODEL, temperature=TEMPERATURE, streaming=True)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response.queries
    queries.append(state.question)
    logger.info(f"Queries: {queries}")
    return {"queries": queries}


async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function resolves the retriever for the current session (the PDF
    tied to this chat, via config.configurable) and uses it to fetch
    relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration identifying which session's
            document collection to retrieve from.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    logger.info("---RETRIEVING DOCUMENTS---")
    logger.info(f"Query for the retrieval process: {state.query}")

    compression_retriever = get_retriever_for_session(config)
    response = compression_retriever.invoke(state.query)

    return {"documents": response}


def retrieve_in_parallel(state: ResearcherState) -> list[Send]:
    """Create parallel retrieval tasks for each generated query.

    This function prepares parallel document retrieval tasks for each query in the researcher's state.

    Args:
        state (ResearcherState): The current state of the researcher, including the generated queries.

    Returns:
        Literal["retrieve_documents"]: A list of Send objects, each representing a document retrieval task.

    Behavior:
        - Creates a Send object for each query in the state.
        - Each Send object targets the "retrieve_documents" node with the corresponding query.
    """
    return [
        Send("retrieve_and_rerank_documents", QueryState(query=query)) for query in state.queries
    ]


builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()
