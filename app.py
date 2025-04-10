from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
from flask_cors import CORS

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.documents import Document

# Load environment & OPENAI_API_KEY
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

# ==== Configuration ====
UPLOAD_FOLDER = './uploads'
VECTOR_FOLDER = './vectorstore'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

# Load API key securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ==== Initialize Flask App ====
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enables CORS in Flask
CORS(app)


# ==== Core Embedding Logic ====
def process_pdf_and_update_embeddings(pdf_path, vector_path, filename):
    # Load PDF pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    # Add filename and page number metadata
    for page_num, doc in enumerate(pages):
        doc.metadata["filename"] = filename
        doc.metadata["page_number"] = page_num + 1

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    # Load or create vector store
    embeddings = OpenAIEmbeddings()
    if os.path.exists(os.path.join(vector_path, "index.faiss")):
        print("Loading existing vector store...")
        vector_store = FAISS.load_local(
                            vector_path, 
                            embeddings,
                            allow_dangerous_deserialization=True
                            )
        vector_store.add_documents(chunks)
    else:
        print("Creating new vector store...")
        vector_store = FAISS.from_documents(chunks, embeddings)

    # Save vector store
    vector_store.save_local(vector_path)


# ==== Embed Endpoint ====
@app.route('/embed', methods=['POST'])
def embed_pdf():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
    vector_path = VECTOR_FOLDER

    try:
        file.save(save_path)
        process_pdf_and_update_embeddings(save_path, vector_path, filename)
        return jsonify({
            "message": "Embedding added to vector store",
            "filename": filename,
            "id": unique_id
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==== Ask Question Endpoint ====
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get("query", "")
    print (query)
    if not query:
        return jsonify({"error": "No query provided"}), 400

    try:
        # Load vector store
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
                            VECTOR_FOLDER, 
                            embeddings,
                            allow_dangerous_deserialization=True
                            )

        # Retrieve top-k documents
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        relevant_docs = retriever.get_relevant_documents(query)

        # Generate citations
        citations = []
        for doc in relevant_docs:
            meta = doc.metadata
            source = {
                "filename": meta.get("filename", "Unknown"),
                "page_number": meta.get("page_number", "N/A"),
                "content_snippet": doc.page_content[:200]
            }
            citations.append(source)

        # Generate answer using LLM with context
        llm = ChatOpenAI(temperature=0)
        chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        result = chain({"input_documents": relevant_docs, "question": query}, return_only_outputs=True)

        return jsonify({
            "question": query,
            "answer": result.get("output_text"),
            "citations": citations,
            "retrieval_count": len(relevant_docs),
            "raw_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in relevant_docs
            ]
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==== Health Check ====
@app.route('/')
def health_check():
    return jsonify({"status": "API is running"}), 200


# ==== Run Server ====
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
