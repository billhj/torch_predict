from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

class Item(BaseModel):
    a: int = None
    b: int = None

@app.post('/test')
def predict(request_data: Item):
    res = {"res": 1}
    return res

@app.get("/")
def read_root():
    return {"hello":"world"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=8080, workers=1)