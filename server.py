"""

IDE: PyCharm
Project: semantic-match-classifier
Author: Robin
Filename: server.py
Date: 22.03.2020
Serves the
"""

import os
from typing import List, Dict

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# init webapp
from data import BertPreprocessor
from model import get_model, get_config_and_device, copy_to_device

app = FastAPI()

# load config and device
config_file = "data/config/bert_cls_config.json"
config, device = get_config_and_device(config_file, cpu_only=True)
max_sequence_length = 50

# init model
preprocessor = BertPreprocessor()
model = get_model("bert_cls_basic", config["model_config"], device)
model.load_state_dict(torch.load("data/models/" + config["model"] + ".pt", map_location=device))
model.eval()


# data models
class Evidence(BaseModel):
    title: str = ""
    text: str = ""
    url: str = ""
    for_class: str = ""


class AnalysisResponse(BaseModel):
    text: str = ""
    classification: Dict = dict()
    evidence: List[Evidence] = list()


class AnalysisRequest(BaseModel):
    text: str = ""

@app.get("/api/info")
def api_info():
    return "CoronaFaktenCheck Model 0.0.1 alpha"


@app.post("/api/analyze", response_model=AnalysisResponse, description="Analyzes news facts and returns findings")
def extract_text(request: AnalysisRequest):
    if request is not None:
        response = AnalysisResponse(text=request.text)

        tokenized = preprocessor.process_text(request.text, None)
        data = copy_to_device(tokenized, ["token_id_tensor", "type_id_tensor", "attn_mask_tensor"], device)

        logits = model(**data)

        probs = torch.softmax(logits, dim=1).tolist()[0]
        labels = config["labels"]

        for i, label in enumerate(labels):
            response.classification[label] = probs[i]

        return response
    raise HTTPException(status_code=400, detail="Invalid request")


if __name__ == "__main__":
    port = os.getenv("PORT", 8000)
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
