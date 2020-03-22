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

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class Evidence(BaseModel):
    title: str = ""
    text: str = ""
    url: str = ""
    for_class: str = ""


class AnalysisResponse(BaseModel):
    text: str = ""
    classification: Dict = dict()
    evidence: List[Evidence] = list()


@app.get("/api/info")
def api_info():
    return "Corona Fakten Check Model v0.1"


@app.post("/api/analyze", response_model=AnalysisResponse, description="Analyzes news facts and returns findings")
def extract_text(text: str, metadata: dict = dict()):
    if text is not None and metadata is not None:
        response = AnalysisResponse()

        return response
    raise HTTPException(status_code=400, detail="Invalid request")


if __name__ == "__main__":
    port = os.getenv("PORT", 8000)
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
