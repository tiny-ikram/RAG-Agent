from fastapi import FastAPI, UploadFile
import shutil
from rag_agent import init_retriever, index_pdf, query_agent

app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_retriever()

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    file_path = f"./data/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    index_pdf(file_path)
    return {"status": "Document indexed", "filename": file.filename}

@app.post("/query")
async def query_endpoint(question: str):
    result = query_agent(question)
    return {
        "question": question,
        "answer": result["response"],
        "sources": {
            "texts": [t.text for t in result["context"]["texts"]],
            "images": [img for img in result["context"]["images"]]
        }
    }
