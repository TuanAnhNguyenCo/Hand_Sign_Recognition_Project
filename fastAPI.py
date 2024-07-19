from fastapi import FastAPI, File, UploadFile, Request
from secrets import token_hex
from vtn_hc_inference import Inference
import requests 
import json
import hashlib

from fastapi.responses import FileResponse
app = FastAPI()
infer = Inference()


@app.get('/')
def root():
    return {'message': 'Welcome to the Image Classification API'}


@app.post("/uploadVideo/")
async def create_upload_file(file: UploadFile):
    print("Received Video")
    file_path = f"receivedVideo/{file.filename}"
    
 
    with open(file_path,'wb') as f:
        content = await file.read()
        f.write(content)
    
    sentence,duration = infer.predict(file_path)
    return {'message': f'Results: {sentence}','duration': duration}



