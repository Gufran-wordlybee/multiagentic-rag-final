FROM python:3.11-slim

# System libraries required by docling's PDF/image pipeline (OpenCV, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model + docling's layout/OCR models at build
# time so the first upload/request in production isn't slow (and doesn't
# fail if the deploy host has restricted egress at runtime).
#
# NOTE: just constructing DocumentConverter() does NOT trigger docling's
# model download — StandardPdfPipeline only fetches its weights lazily,
# inside the pipeline object created on the first real convert() call. So
# priming it here means actually calling the same download function the
# pipeline uses, and this step is NOT allowed to fail silently: if it
# fails at build time, it will instead fail (or hang, on a slow host) on
# a real user's first upload in production.
#
# download_models_hf() downloads to a fixed local folder and returns that
# path; we capture it to a file so retriever.py can point PdfPipelineOptions
# at it explicitly at runtime via DOCLING_ARTIFACTS_PATH, instead of relying
# on docling's default ~/.cache resolving to the same place at build time
# and runtime (which happens to hold here since neither stage sets USER,
# but shouldn't be depended on implicitly).
RUN python -c "from langchain_community.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')"
RUN python -c "\
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline; \
path = StandardPdfPipeline.download_models_hf(); \
open('/app/.docling_artifacts_path', 'w').write(str(path))"
ENV DOCLING_ARTIFACTS_PATH_FILE=/app/.docling_artifacts_path

COPY . .

# Session data (uploaded PDFs + their Chroma indexes) lives here at runtime.
# Mount a volume at this path if you need sessions to survive restarts;
# otherwise sessions are ephemeral, which is fine for a stateless deploy.
ENV SESSIONS_DIR=/app/sessions
RUN mkdir -p /app/sessions

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
