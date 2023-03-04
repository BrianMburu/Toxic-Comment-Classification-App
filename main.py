import time
import uvicorn
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel,Field
from transformers import DistilBertTokenizer
from utils import predict_text, clean_text, DistilBERTClass, DEVICE

app = FastAPI()

# Load the trained model
model = DistilBERTClass()
model.load_state_dict(torch.load("model/toxic_comment.pkl", map_location=torch.device('cpu')))
model.to(DEVICE)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)

class InputText(BaseModel):
    text: str = Field(...)
    stemmer: str = Field(...)

# Define the prediction endpoint
@app.post('/predict')
def predict(input_text: InputText):
    text = input_text.text
    stemmer = input_text.stemmer
    """
    Route to the prediction endpoint
    """
    start_time = time.monotonic()
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text length exceeds 2000 characters")
    
    cleaned_text = clean_text(text, stemmer)
   
    if len(cleaned_text)==0:
        raise HTTPException(status_code=400, detail="Phrase stemmed/lemmatized to zero characters")
    
    results = predict_text(cleaned_text, model, tokenizer)
    
    end_time = time.monotonic()
    duration = end_time - start_time
    # Log the runtime
    if duration > 5:
        raise HTTPException(status_code=500, detail=f"Prediction took too long ({duration:.2f} seconds)")
    
    return {'predictions': results,'Execution Time': duration}

