from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from rag_answer import generate_consolidated_answer

app = FastAPI()

class QuestionRequest(BaseModel):
    questions: List[str]

@app.post("/api/rag_query")
async def rag_query(request: QuestionRequest):
    try:
        answer = generate_consolidated_answer(request.questions)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
