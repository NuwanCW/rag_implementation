import os
import time
import logging
import pyinotify  # for inotify-based file watching
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Sentence-BERT model and Chromadb client
model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.HttpClient(host="chromadb-container", port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))

# Create or get the "documents" collection
collection = client.get_or_create_collection("documents")

# Directory to watch
WATCH_DIRECTORY = "/app/source_documents"  # This directory should be mapped via Docker volumes

# Function to extract text from the document
def process_new_document(file_path):
    """Process a new document and add it to Chromadb."""
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.warning(f"File does not exist: {file_path}")
            return

        # Read the document's content
        with open(file_path, 'r') as file:
            document_text = file.read()
            logger.info(f"Read document length: {len(document_text)}")
            logger.debug(f"Document text (first 100 chars): {document_text[:100]}")
        
        if document_text.strip():  # Only process non-empty documents
            # Create embeddings for the document
            embeddings = model.encode([document_text])

            # Generate a unique document ID (could be based on file name or UUID)
            document_id = os.path.basename(file_path)

            # Check if the document already exists and delete it if necessary
            existing_documents = collection.get(ids=[document_id])
            if existing_documents:
                collection.delete(ids=[document_id])
                logger.info(f"Document {document_id} replaced in Chromadb.")

            # Store the document in Chromadb
            collection.add(documents=[document_text], embeddings=[embeddings[0]], metadatas=[{"source": file_path}], ids=[document_id])
            logger.info(f"Document {document_id} added to Chromadb.")
        else:
            logger.warning(f"The file {file_path} is empty or contains only whitespace.")

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
