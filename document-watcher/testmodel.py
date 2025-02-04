from flask import Flask, request, jsonify
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import requests
import logging
import json
from langchain.llms.base import LLM
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Initialize Flask app
app = Flask(__name__)

# Sentence-BERT model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# ChromaDB HTTP Client
from chromadb.config import Settings
import chromadb

chroma_client = chromadb.HttpClient(
    host="chromadb-container",
    port=8000,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
)

# Llama API configuration
ollama_host = "http://ollama-container:11434"

# Logging setup
logging.basicConfig(level=logging.INFO)

# Define a custom LLM class for LangChain
class CustomLlamaLLM(LLM):
    def __init__(self, api_url: str=f"{ollama_host}/api/generate"):
        self.api_url = api_url
    def _call(self, prompt: str, stop=None) -> str:
        try:
            payload = {
                "model": "llama3:latest",
                "prompt": prompt,
            }
            response = requests.post(self.api_url, json=payload, stream=True)
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            result = json.loads(line.decode('utf-8'))
                            full_response += result.get("response", "")
                            if result.get("done", False):
                                break
                        except json.JSONDecodeError as e:
                            logging.error(f"JSON decode error: {e}")
                            continue
                return full_response.strip()
            else:
                logging.error(f"Ollama API error: {response.status_code}, {response.text}")
                return "Error: Unable to generate response."
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error: {e}")
            return "Error: Unable to reach Ollama API."
    @property
    def _identifying_params(self):
        return {"api_url": self.api_url}
    @property
    def _llm_type(self) -> str:
        return "custom_llama_llm"

# Initialize custom Llama LLM
llama_llm = CustomLlamaLLM(api_url=f"{ollama_host}/api/generate")
