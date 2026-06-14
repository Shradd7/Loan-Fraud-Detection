from pathlib import Path
import json
import sys
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ficoforce.config import DATA_PATH, ID_COLUMN, METRICS_PATH, MODEL_PATH
from ficoforce.agents import run_location_agents
from ficoforce.predict import ModelNotTrainedError, load_model, predict_records
from ficoforce.rag import load_location_records, verify_location


app = FastAPI(
    title="FICOFORCE Risk Intelligence API",
    description="ML fraud-risk scoring and local RAG-based borrower location verification.",
    version="2026.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_ARTIFACT = None


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class LocationRequest(BaseModel):
    account_id: str | None = None
    record: dict[str, str] | None = None


class PredictByIdRequest(BaseModel):
    unique_id: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_trained": MODEL_PATH.exists(),
        "location_records": len(load_location_records()),
    }


@app.get("/model-info")
def model_info():
    metrics = None
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {
        "model_path": str(MODEL_PATH),
        "model_trained": MODEL_PATH.exists(),
        "metrics_json": metrics,
    }


@app.post("/predict")
def predict(request: PredictRequest):
    global _MODEL_ARTIFACT
    try:
        if _MODEL_ARTIFACT is None:
            _MODEL_ARTIFACT = load_model()
        return {"results": predict_records(request.records, _MODEL_ARTIFACT)}
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict-by-id")
def predict_by_id(request: PredictByIdRequest):
    record = _find_credit_record(request.unique_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"UNIQUE_ID {request.unique_id} was not found in the Task 1 training data.",
        )
    result = predict(PredictRequest(records=[record]))["results"][0]
    return {"unique_id": request.unique_id, **result}


@app.post("/location-verify")
def location_verify(request: LocationRequest):
    if request.record:
        return _location_response(request.record)
    if request.account_id:
        for record in load_location_records():
            if str(record.get("Acc.No", "")).strip() == str(request.account_id).strip():
                return _location_response(record)
        raise HTTPException(status_code=404, detail="Account ID not found in dummy Task 2 data.")
    raise HTTPException(status_code=400, detail="Provide either account_id or record.")


@app.get("/location-records")
def location_records(limit: int = 25):
    rows = load_location_records()
    return {"records": rows[: max(limit, 0)], "total": len(rows)}


def _location_response(record: dict[str, str]):
    rag_result = verify_location(record)
    agent_result = run_location_agents(record)
    return {
        **rag_result,
        "pipeline": "local_multi_agent_rag",
        "retrieval_plan": agent_result["planner"],
        "agent_outputs": agent_result["agents"],
        "agent_final": agent_result["final"],
    }


def _find_credit_record(unique_id: str) -> dict[str, Any] | None:
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Task 1 data file is missing.")

    target = str(unique_id).strip()
    for chunk in pd.read_csv(DATA_PATH, chunksize=50000):
        matches = chunk[chunk[ID_COLUMN].astype(str) == target]
        if not matches.empty:
            row = matches.iloc[0].drop(labels=[ID_COLUMN, "TARGET"], errors="ignore")
            return row.where(pd.notna(row), None).to_dict()
    return None
