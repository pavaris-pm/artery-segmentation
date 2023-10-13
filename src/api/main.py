# main.py
import uvicorn
from fastapi import FastAPI, File, UploadFile
from utils import read_imagefile
from models.segformer.core import segment

app = FastAPI()

@app.get('/index')
async def hello_world():
    return "test api"

@app.post("/predict/image")
async def prediction_api(
    file: UploadFile = File(...)
    ):
    # read an image uploaded by user
    image = read_imagefile(await file.read())
    # make a prediction
    output = segment(image)
    return output


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')