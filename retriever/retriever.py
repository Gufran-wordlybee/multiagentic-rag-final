# from docling.document_converter import DocumentConverter , mac gpu to cpu
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

from langchain_text_splitters import MarkdownHeaderTextSplitter
from utils.utils import config

from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
#since openai is not free
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.retrievers import EnsembleRetriever, BM25Retriever
from typing import List, Any
import logging
import os
from dotenv import load_dotenv
import rank_bm25

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_docling_artifacts_path() -> str | None:
    """Resolve a pre-downloaded docling model artifacts directory, if any.

    The Dockerfile primes docling's models at build time via
    StandardPdfPipeline.download_models_hf() and writes the resulting path
    to DOCLING_ARTIFACTS_PATH_FILE. Reading it back here and passing it to
    PdfPipelineOptions explicitly means the runtime DocumentConverter finds
    those baked-in weights directly, rather than relying on docling's
    default ~/.cache resolution happening to match between the build stage
    and the running container (true today since neither runs as a non-root
    USER, but not something to depend on implicitly). If the file/env var
    isn't set (e.g. local dev without Docker), this returns None and
    docling falls back to its normal default-cache / lazy-download behavior.
    """
    direct = os.environ.get("DOCLING_ARTIFACTS_PATH")
    if direct:
        return direct

    path_file = os.environ.get("DOCLING_ARTIFACTS_PATH_FILE")
    if path_file and os.path.exists(path_file):
        try:
            with open(path_file, "r") as f:
                resolved = f.read().strip()
                return resolved or None
        except OSError:
            return None
    return None


class DocumentProcessor:
    """
    Handles document conversion and splitting.
    """
    def __init__(self, headers_to_split_on: List[str]):
        self.headers_to_split_on = headers_to_split_on

    def process(self, source: Any) -> List[str]:
        """
        Converts a document to markdown and splits it into chunks.

        Args:
            source (Any): The source document to process.

        Returns:
            List[str]: List of document sections split by headers.
        """
        try:
            logger.info("Starting document processing.")
            # converter = DocumentConverter()
            """ 
            my error : 
            This is why Docling is automatically selecting:
            Accelerator device: 'mps'
            We need to tell it "Don't use MPS, use CPU."
            """
            pipeline_options = PdfPipelineOptions()

            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CPU
            )

            artifacts_path = _resolve_docling_artifacts_path()
            if artifacts_path:
                pipeline_options.artifacts_path = artifacts_path
                logger.info(f"Using pre-downloaded docling artifacts at '{artifacts_path}'.")

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )

            markdown_document = converter.convert(source).document.export_to_markdown()
            markdown_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)
            docs_list = markdown_splitter.split_text(markdown_document)
            logger.info("Document processed successfully.")
            return docs_list
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise RuntimeError(f"Error processing document: {e}")


class IndexBuilder:
    """
    Builds vector-based and BM25-based retrievers.
    """
    def __init__(self, docs_list: List[str], collection_name: str, persist_directory: str, load_documents: bool):
        self.docs_list = docs_list
        self.collection_name = collection_name
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.load_documents = load_documents

    def build_vectorstore(self):
        """
        Initializes the Chroma vectorstore with the provided documents and embeddings.
        """
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )
        try:
            logger.info("Building vectorstore.")
            self.vectorstore = Chroma.from_documents(
                persist_directory=self.persist_directory,
                documents=self.docs_list,
                collection_name=self.collection_name,
                embedding=embeddings,
            )         
            logger.info("Vectorstore built successfully.")
        except Exception as e:
            logger.error(f"Error building vectorstore: {e}")
            raise RuntimeError(f"Error building vectorstore: {e}")

    def build_retrievers(self):
        """
        Builds BM25 and vector-based retrievers and combines them into an ensemble retriever.

        Returns:
            EnsembleRetriever: Combined retriever using BM25 and vector-based methods.
        """
        try:
            logger.info("Building BM25 retriever.")
            bm25_retriever = BM25Retriever.from_documents(self.docs_list, search_kwargs={"k": 4})

            logger.info("Building vector-based retrievers.")
            retriever_vanilla = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            retriever_mmr = self.vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 4}
            )

            logger.info("Combining retrievers into an ensemble retriever.")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_vanilla, retriever_mmr, bm25_retriever],
                weights=[0.3, 0.3, 0.4],
            )
            logger.info("Retrievers built successfully.")
            return ensemble_retriever
        except Exception as e:
            logger.error(f"Error building retrievers: {e}")
            raise RuntimeError(f"Error building retrievers: {e}")


def index_pdf(
    filepath: str,
    collection_name: str,
    persist_directory: str,
    headers_to_split_on: List[Any] = None,
):
    """Process a single PDF and build (or rebuild) its Chroma vectorstore.

    This is the reusable entrypoint used by the web server to index a
    freshly-uploaded PDF into its own isolated collection/session directory.
    It performs the same PDF -> Markdown -> header-chunked pipeline as the
    original CLI flow, just parameterized instead of driven by config.yaml.

    Args:
        filepath: Path to the PDF file on disk.
        collection_name: Name of the Chroma collection to create/use.
        persist_directory: Directory where the Chroma collection is persisted
            (should be unique per session so uploads never mix).
        headers_to_split_on: Markdown header splitter config. Defaults to
            the standard H1/H2 split used elsewhere in this project.

    Returns:
        Chroma: The persisted vectorstore instance for this document.
    """
    if headers_to_split_on is None:
        headers_to_split_on = config["retriever"]["headers_to_split_on"]

    logger.info(f"Indexing PDF '{filepath}' into collection '{collection_name}'.")
    processor = DocumentProcessor(headers_to_split_on)
    docs_list = processor.process(filepath)
    logger.info(f"{len(docs_list)} chunks generated from '{filepath}'.")

    if not docs_list:
        raise RuntimeError(
            "No content could be extracted from the PDF. It may be empty, "
            "image-only/scanned, or corrupted."
        )

    index_builder = IndexBuilder(
        docs_list,
        collection_name,
        persist_directory=persist_directory,
        load_documents=True,
    )
    index_builder.build_vectorstore()
    logger.info(f"Vectorstore for collection '{collection_name}' ready at '{persist_directory}'.")
    return index_builder.vectorstore


if __name__ == "__main__":
    # Standalone indexing run: builds a local vector_db/manual-index collection
    # from config.yaml's retriever.file. Mainly useful for testing this module
    # in isolation; app.py and server.py call index_pdf() directly instead.
    filepath = config["retriever"]["file"]
    print(f"Retriever entry — indexing '{filepath}'")
    try:
        index_pdf(
            filepath=filepath,
            collection_name="manual-index",
            persist_directory="vector_db",
        )
        logger.info("Index built successfully. Ready for use.")
    except RuntimeError as e:
        logger.critical(f"Failed to build index: {e}")
        exit(1)
