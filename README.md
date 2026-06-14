# FICOFORCE

FICOFORCE is a loan fraud detection and review system. It combines:

- Task 1: ML-based fraud risk detection with a reproducible LightGBM training pipeline
- Task 2: local RAG-based borrower location verification without Groq, Selenium, or scraping
- FastAPI backend for model scoring and verification
- React analyst dashboard for credit review workflows

The project is structured as an end-to-end Data Scientist / Product Analyst case study for financial risk teams.

## Current Results

The full-data fast SMOTE-Tomek experiment produced:

| Metric | Score |
|---|---:|
| F1 | 0.596 |
| Precision | 0.546 |
| Recall | 0.656 |
| Accuracy | 0.904 |
| PR-AUC | 0.631 |
| ROC-AUC | 0.920 |
| Tuned threshold | 0.730 |

The trained model artifact is stored locally in `artifacts/` and is not committed to Git.

## Architecture

```text
Task 1 data
  -> feature engineering
  -> LightGBM training pipeline
  -> threshold tuning
  -> FastAPI /predict and /predict-by-id
  -> React risk panel

Task 2 dummy records
  -> local RAG retrieval
  -> planner agent
  -> static identity verifier
  -> activity location verifier
  -> conflict resolver and scorer
  -> FastAPI /location-verify
  -> React evidence panel
```

## Project Structure

```text
Loan-Fraud-Detection/
  backend/                  FastAPI service
  frontend/                 React analyst dashboard
  src/ficoforce/            Training, prediction, feature, and RAG code
  data/rag/                 Local retrieval knowledge base
  docs/                     Model card and data card
  tests/                    Focused unit tests
  assets/                   Existing presentation images
  Task_1_*.ipynb            Original hackathon notebook
  Task_2_*.ipynb            Original hackathon notebook
  HACKATHON_TRAINING_DATA.CSV
  Dummy Dataset Final.txt
```

Large data/model artifacts are ignored by Git.

## Data Policy

Raw datasets are intentionally not committed:

- `HACKATHON_TRAINING_DATA.CSV`
- `Dummy Dataset Final.txt`
- `artifacts/`

Place these files locally before training or running the full demo. Public GitHub should contain code, docs, model cards, tests, and the local RAG knowledge base only.

## System Overview

### Task 1: ML Fraud Risk Detection

The notebook workflow is available as a reproducible training package:

- deterministic feature engineering
- Y/N flag encoding
- duration parsing
- income band handling
- overspend ratio and consecutive overspend features
- outstanding balance and debit trend slopes
- fast profile using imputation plus LightGBM class weighting
- optional deep profile with Yeo-Johnson transformation
- optional SMOTE-Tomek imbalance experiment for smaller samples
- LightGBM classifier
- validation-set threshold tuning for F1
- test-set reporting with F1, precision, recall, accuracy, PR-AUC, ROC-AUC, and confusion matrix
- business-cost reporting for false positives and false negatives
- segment-audit utility for fairness-style checks across product, income, or geography groups

### Task 2: Location Verification

Location verification uses a local RAG system:

- no API key required
- no browser automation
- no LinkedIn scraping
- retrieves branch/state/location rules from `data/rag/location_knowledge.json`
- scores evidence from branch code, address, UPI location, frequent location, last location, driving license, vehicle number, and phone prefix
- runs a local 4-agent pipeline: planner, static verifier, activity verifier, and final conflict scorer
- returns confidence, conflict count, manual review flag, and evidence trail

## Setup

Create a local environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

If Windows opens the Microsoft Store for `python`, install Python 3.10+ from python.org and reopen the terminal.

## Train Task 1

Quick smoke training on a sample:

```bash
python -m ficoforce.train --sample-frac 0.05
```

Full training, recommended first:

```bash
python -m ficoforce.train
```

SMOTE-Tomek experiment on a smaller sample:

```bash
python -m ficoforce.train --profile deep --resampling smote_tomek --sample-frac 0.25
```

Full SMOTE-Tomek run with progress logs:

```bash
python -m ficoforce.train --profile fast --resampling smote_tomek
```

Use `--quiet` only if you want to hide the step logs.

Outputs:

```text
artifacts/ficoforce_default_model.joblib
artifacts/ficoforce_metrics.json
```

The model is ignored by Git, which is correct for a GitHub-connected project.

## Run Backend

```bash
uvicorn backend.main:app --reload
```

API:

- `GET /health`
- `GET /model-info`
- `POST /predict`
- `POST /predict-by-id`
- `POST /location-verify`
- `GET /location-records`

## Run Frontend

```bash
cd frontend
npm.cmd install
npm.cmd start
```

Open:

```text
http://localhost:3000
```

The dashboard talks to:

```text
http://localhost:8000
```

## Docker

```bash
docker-compose up --build
```

Backend:

```text
http://localhost:8000
```

Frontend:

```text
http://localhost:3000
```

## Testing

```bash
pip install -r requirements.txt
pytest -q
```

Frontend:

```bash
cd frontend
npm.cmd test -- --watchAll=false
```

## Demo Inputs

Task 1 uses `UNIQUE_ID` from the training CSV:

| UNIQUE_ID | Expected output |
|---|---|
| `2032` | low fraud probability |
| `2047` | medium fraud probability |
| `5558` | high fraud probability / Fraud Risk |

Task 2 uses account IDs from the local dummy dataset:

| Account ID | Expected output |
|---|---|
| `340128` | medium-confidence Pune/Maharashtra case |
| `123456` | high-confidence Jaipur/Rajasthan case with conflicts |
| `145932` | low-confidence manual-review case |

## Portfolio Talking Points

- I converted a notebook-only hackathon solution into a reproducible ML system.
- I tuned the decision threshold for F1 instead of relying on a default 0.5 cutoff.
- I added PR-AUC and ROC-AUC because fraud/default data is imbalanced.
- I replaced paid/API-dependent agents with local RAG for auditability.
- I added model/data cards and a human-review policy for responsible AI.
- I built a dashboard that supports analyst decisions rather than only showing a binary label.

## Current Limitations

- Task 2 uses synthetic dummy data and local rules, so it is suitable for demos but not real identity verification.
- Fairness checks across sensitive or proxy groups should be added before any real lending use.

## Author

Shraddhan Singhai
