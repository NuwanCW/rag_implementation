from flask import Flask, request, jsonify
from pydantic import BaseModel
import logging
import json
import chromadb
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.llms.base import LLM
from pydantic import Field
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# Llama model setup
local_llm = "llama3:latest"
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
llm = ChatOllama(model=local_llm, temperature=0)
chroma_client = chromadb.HttpClient(
    host="chroma-server",
    port=8000,
    settings=Settings(allow_reset=True, anonymized_telemetry=False)
)
embedding_function = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
vectorstore = Chroma(
    client=chroma_client,  # Use your ChromaDB client
    collection_name="documents",  # Change as needed
    embedding_function=embedding_function,
)

router_instructions = """You are an expert at routing a user question to a vectorstore.
The vectorstore contains documents related to Bible studies and messages.
Use the vectorstore for questions on these topics.
Return JSON with single key, datasource, that is 'vectorstore'."""
# Doc grader instructions
doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""
# Grader prompt
doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 
This carefully and objectively assess whether the document contains at least some information that is relevant to the question.
Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""
rag_prompt = """You are an assistant for question-answering tasks. 
Here is the context to use to answer the question:
{context} 
Think carefully about the above context. 
Now, review the user question:
{question}
Provide an answer to this questions using only the above context. 
Use three sentences maximum and keep the answer concise.
Answer:"""
hallucination_grader_instructions = """
You are a teacher grading a quiz. 
You will be given FACTS and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.
Score:
A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""
# Grader prompt
hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 
Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""
# Answer grader instructions
answer_grader_instructions = """You are a teacher grading a quiz. 
You will be given a QUESTION and a STUDENT ANSWER. 
Here is the grade criteria to follow:
(1) The STUDENT ANSWER helps to answer the QUESTION
Score:
A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 
The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.
A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""
# Grader prompt
answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 
Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)

qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=compression_retriever,
            chain_type="refine"
        )
# Initialize Flask app

# Request schema for query input
class QueryRequest(BaseModel):
    question: str
    # top_k: int = 1
    
# Define route for query input
@app.route("/query", methods=["POST"])
def query_system():
    try:
        # question = "what happend to Joseph?"
        # Parse incoming JSON data
        data = request.get_json()
        question = QueryRequest(**data)
        print(question.question)
     
        response = qa_chain({"query": question.question})
# print(result)        
        return jsonify({"response": response}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9005, debug=True)


