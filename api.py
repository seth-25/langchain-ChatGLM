from configs.model_config import *
from chains.local_doc_qa import LocalDocQA

import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from starlette.responses import RedirectResponse

app = FastAPI()

global local_doc_qa, vs_path


# return top-k text chunk from vector store
VECTOR_SEARCH_TOP_K = 10

# LLM input history length
LLM_HISTORY_LEN = 3

# Show reply with source text from input document
REPLY_WITH_SOURCE = True

class Query(BaseModel):
    query: str

@app.get('/')
async def document():
    return RedirectResponse(url="/docs")

@app.on_event("startup")
async def get_local_doc_qa():
    global local_doc_qa
    local_doc_qa = LocalDocQA()
    local_doc_qa.init_cfg(top_k=3)

@app.post("/qa")
async def get_answer(UserQuery: Query):
    response = {
        "status": 0,
        "message": "",
        "answer": None
    }
    global vs_path
    history = []
    try:
        resp, history = local_doc_qa.get_knowledge_based_answer(query=UserQuery.query,
                                                                chat_history=history)
        if REPLY_WITH_SOURCE:
            response["answer"] = resp
        else:
            response['answer'] = resp["result"]
        
        response["message"] = 'successful'
        response["status"] = 1

    except Exception as err:
        print(err)
        response["message"] = err
        
    return response


if __name__ == "__main__":
    uvicorn.run(
        app='api:app', 
        host='0.0.0.0', 
        port=8100,
        reload = True,
        )

