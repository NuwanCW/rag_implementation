ARG CHROMA_VERSION=latest
FROM ghcr.io/chroma-core/chroma:${CHROMA_VERSION} as base
RUN pip install openfga-sdk
COPY chromadb/chroma_auth/ /chroma/chroma_auth
# COPY chroma_auth/instr/__init__.py /chroma/chromadb/server/fastapi/__init__.py
