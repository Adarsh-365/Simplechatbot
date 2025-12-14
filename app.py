from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import os
import shutil
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Tuple
import json
import time

load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

SESSIONS_DIR = "ragbot/sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

db = None
faiss_retriever = None
bm25_retriever = None
docs = []


def get_documents_from_file(file_path: str):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif file_extension in [".txt", ".md"]:
        loader = TextLoader(file_path)
    else:
        return None
    return loader.load()

def manual_rerank(results: List[Tuple[str, List[Document]]], k: int = 5) -> List[Document]:
    if not results:
        return []

    all_docs = {}
    for retriever_name, docs in results:
        for i, doc in enumerate(docs):
            if doc.page_content not in all_docs:
                all_docs[doc.page_content] = {
                    "doc": doc,
                    "scores": {},
                    "ranks": {}
                }
            all_docs[doc.page_content]["ranks"][retriever_name] = i + 1

    for content, data in all_docs.items():
        total_score = 0
        for retriever_name, rank in data["ranks"].items():
            total_score += 1 / (60 + rank)
        data["scores"]["rrf"] = total_score

    sorted_docs = sorted(all_docs.values(), key=lambda x: x["scores"]["rrf"], reverse=True)
    return [d["doc"] for d in sorted_docs[:k]]


def manual_retriever(query: str) -> List[Document]:
    bm25_docs = bm25_retriever.get_relevant_documents(query)
    faiss_docs = faiss_retriever.invoke(query)
    
    reranked_docs = manual_rerank([
        ("bm25", bm25_docs),
        ("faiss", faiss_docs)
    ])
    return reranked_docs


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    global db, faiss_retriever, bm25_retriever, docs
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    documents = get_documents_from_file(file_path)
    if not documents:
        return {"error": "Unsupported file format."}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    model_name = "all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embeddings = hf
    
    db = FAISS.from_documents(docs, embeddings)
    faiss_retriever = db.as_retriever()

    tokenized_docs = [doc.page_content.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)

    class BM25Retriever:
        def __init__(self, bm25_index, docs):
            self.bm25_index = bm25_index
            self.docs = docs

        def get_relevant_documents(self, query: str):
            tokenized_query = query.split()
            doc_scores = self.bm25_index.get_scores(tokenized_query)
            k = 5
            top_n_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:k]
            return [self.docs[i] for i in top_n_indices]

    bm25_retriever = BM25Retriever(bm25, docs)

    return {"message": f"File '{file.filename}' processed successfully."}


@app.post("/query/")
async def query(query: str = Form(...), session_id: str = Form(None), api_key: str = Form(None), model: str = Form("gemini-1.5-flash")):
    if not faiss_retriever:
        return {"error": "Please upload a document first."}

    if not session_id:
        session_id = str(int(time.time()))

    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")

    try:
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
            
        llm = ChatGoogleGenerativeAI(model=model)
        
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                chat_history_list = json.load(f)
        else:
            chat_history_list = []

        formatted_chat_history = ""
        for message in chat_history_list:
            formatted_chat_history += f"{message['sender'].capitalize()}: {message['text']}\n"

        prompt_template = """
        Answer the following question based only on the provided context and the conversation history.
        Think step by step before providing a detailed answer.

        Conversation History:
        {chat_history}

        <context>
        {context}
        </context>
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        chain = (
            {"context": manual_retriever, "question": RunnablePassthrough(), "chat_history": lambda x: formatted_chat_history}
            | prompt
            | llm
            | StrOutputParser()
        )

        response_text = chain.invoke(query)
        answer = response_text

        chat_history_list.append({"sender": "user", "text": query})
        chat_history_list.append({"sender": "bot", "text": answer})
        with open(session_file, "w") as f:
            json.dump(chat_history_list, f)

        return {"answer": answer, "session_id": session_id}
    except Exception as e:
        return {"error": str(e)}

@app.get("/sessions", response_class=JSONResponse)
async def get_sessions():
    sessions = [f.replace(".json", "") for f in os.listdir(SESSIONS_DIR) if f.endswith(".json")]
    return sorted(sessions, reverse=True)

@app.get("/session/{session_id}", response_class=JSONResponse)
async def get_session(session_id: str):
    session_file = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(session_file):
        with open(session_file, "r") as f:
            return json.load(f)
    return []


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("ragbot/index.html", "r") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
