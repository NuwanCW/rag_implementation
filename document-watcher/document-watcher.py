from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQA
from langchain.document_loaders import TextLoader
import os
import time
import logging
import pyinotify  # for inotify-based file watching
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
client = chromadb.HttpClient(host="chroma-server", port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
# client = chromadb.HttpClient(
#     settings=Settings(chroma_client_auth_provider="chromadb.auth.basic.BasicAuthClientProvider",chroma_client_auth_credentials="admin:password123"))
print(client.heartbeat())  # this should work with or without authentication - it is a public endpoint
# client.get_or_create_collection("test_collection")  # this is a protected endpoint and requires authentication
# client.list_collections()  # this is a protected endpoint and requires authentication

# Create or get the "documents" collection
collection = client.get_or_create_collection("documents")

# Directory to watch
WATCH_DIRECTORY = "/app/source_documents"  # This directory should be mapped via Docker volumes


def process_new_document(file_path):
    """Process a new document and add it to Chromadb."""
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=256, chunk_overlap=40)
        doc_splits = text_splitter.split_documents(documents)
# embedding_model.embed_documents([doc_splits[0].page_content])
        # Read the document's content
        for i, doc_chunk in enumerate(doc_splits):
            chunk_text = doc_chunk.page_content
            chunk_id = f"{os.path.basename(file_path)}_chunk_{i}"

            # Generate embeddings
            embeddings=embedding_model.embed_documents([doc_chunk.page_content])[0]
            # embeddings = embedding_model.embed_documents([chunk_text])[0]

            existing_documents = collection.get(ids=[chunk_id])
            if existing_documents:
                collection.delete(ids=[chunk_id])
                logger.info(f"Document {chunk_id} replaced in Chromadb.")
            # Store in ChromaDB
            collection.add(
                documents=[chunk_text],
                embeddings=[embeddings],
                metadatas=[{"source": file_path, "chunk_index": i}],
                ids=[chunk_id]
            )
            logger.info(f"Processed {len(doc_splits)} chunks from {file_path} into ChromaDB.")
        # else:
        #     logger.warning(f"The file {file_path} is empty or contains only whitespace.")

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")

# Inotify event handler class
class EventHandler(pyinotify.ProcessEvent):
    def process_IN_CREATE(self, event):
        """Triggered when a new file is created."""
        file_path = event.pathname
        logger.info(f"New file detected: {file_path}")
        process_new_document(file_path)

    def process_IN_MODIFY(self, event):
        """Triggered when a file is modified."""
        file_path = event.pathname
        logger.info(f"File modified: {file_path}")
        process_new_document(file_path)

def monitor_directory():
    """Monitor the directory for file events using pyinotify."""
    logger.info("Starting to monitor directory...")
    
    wm = pyinotify.WatchManager()  # Create a WatchManager instance
    handler = EventHandler()  # Instantiate the event handler
    notifier = pyinotify.Notifier(wm, handler)  # Create a Notifier instance
    
    # Add watch for the directory (IN_CREATE event when files are created)
    wm.add_watch(WATCH_DIRECTORY, pyinotify.IN_CREATE | pyinotify.IN_MODIFY)

    # Start monitoring the directory
    notifier.loop()

if __name__ == "__main__":
    monitor_directory()
