from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "HACKATHON_TRAINING_DATA.CSV"
LOCATION_DATA_PATH = PROJECT_ROOT / "Dummy Dataset Final.txt"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "ficoforce_default_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "ficoforce_metrics.json"
RAG_KB_PATH = PROJECT_ROOT / "data" / "rag" / "location_knowledge.json"

TARGET_COLUMN = "TARGET"
ID_COLUMN = "UNIQUE_ID"
