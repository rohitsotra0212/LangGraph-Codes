from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()

saved_items = []

class AnswerSchema(BaseModel):
    query: str
    route: str
    answer: str

@app.get("/")
def home():
    return {"message": "FastAPI is running"}

@app.post("/save")
def save(data: AnswerSchema):
    item = data.model_dump()
    saved_items.append(item)
    return {
        "status": "success",
        "message": "Saved successfully",
        "data": item,
        "total_saved": len(saved_items)
    }

@app.get("/all")
def get_all():
    return {
        "count": len(saved_items),
        "items": saved_items
    }

if __name__ == "__main__":
    uvicorn.run("fast_payload:app", host="127.0.0.1", port=8000, reload=True)