import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import LOCATION_DATA_PATH, RAG_KB_PATH


STATE_BY_CODE = {
    "AP": "Andhra Pradesh",
    "BR": "Bihar",
    "DL": "Delhi",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "KA": "Karnataka",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "TN": "Tamil Nadu",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

CITY_STATE_HINTS = {
    "Ahmedabad": "Gujarat",
    "Amaravati": "Andhra Pradesh",
    "Bareilly": "Uttar Pradesh",
    "Bathinda": "Punjab",
    "Begusarai": "Bihar",
    "Bhilwara": "Rajasthan",
    "Gandhinagar": "Gujarat",
    "GandhiNagar Jaipur": "Rajasthan",
    "Guwahati": "Assam",
    "Gulbarga": "Karnataka",
    "Jaipur": "Rajasthan",
    "Jabalpur": "Madhya Pradesh",
    "Kolkata": "West Bengal",
    "Ludhiana": "Punjab",
    "Mumbai": "Maharashtra",
    "Nagpur": "Maharashtra",
    "Navi Mumbai": "Maharashtra",
    "Noida": "Uttar Pradesh",
    "Pune": "Maharashtra",
    "Rajasthan": "Rajasthan",
    "Salem": "Tamil Nadu",
    "Satara": "Maharashtra",
}


@dataclass
class Evidence:
    source: str
    location: str
    state: str
    weight: float
    reason: str


def load_location_records(path: Path = LOCATION_DATA_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        parsed = [row for row in reader if row]
    if not parsed:
        return rows

    header = _split_wrapped_row(parsed[0])
    for row in parsed[1:]:
        values = _split_wrapped_row(row)
        if len(values) < len(header):
            values += [""] * (len(header) - len(values))
        rows.append({header[i].strip(): values[i].strip() for i in range(len(header))})
    return rows


def retrieve_knowledge(query: str, kb_path: Path = RAG_KB_PATH, top_k: int = 5) -> list[dict[str, Any]]:
    if not kb_path.exists():
        return []
    docs = json.loads(kb_path.read_text(encoding="utf-8"))
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for doc in docs:
        text = json.dumps(doc).lower()
        score = sum(1 for term in terms if term in text)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def verify_location(record: dict[str, str]) -> dict[str, Any]:
    query = " ".join(str(value) for value in record.values())
    retrieved = retrieve_knowledge(query)
    evidence = _extract_evidence(record, retrieved)

    state_scores = Counter()
    for item in evidence:
        state_scores[item.state] += item.weight

    predicted_state = state_scores.most_common(1)[0][0] if state_scores else "Unknown"
    predicted_city = _best_city_for_state(evidence, predicted_state)
    total_score = float(sum(max(item.weight, 0) for item in evidence))
    conflict_count = len({item.state for item in evidence if item.state and item.state != predicted_state})
    adjusted_score = max(total_score - conflict_count * 1.5, 0)

    confidence = "High" if adjusted_score >= 7 else "Medium" if adjusted_score >= 4 else "Low"
    manual_review = confidence == "Low" or conflict_count >= 3

    return {
        "account_id": record.get("Acc.No", ""),
        "name": " ".join(
            part
            for part in [
                record.get("First_Name_Middle_Name", ""),
                record.get("Last_Name", record.get("Last_Name ", "")),
            ]
            if part
        ).strip(),
        "predicted_location": predicted_city,
        "predicted_state": predicted_state,
        "confidence": confidence,
        "score": round(adjusted_score, 2),
        "manual_review": manual_review,
        "conflict_count": conflict_count,
        "retrieved_context": retrieved,
        "evidence": [asdict(item) for item in evidence],
    }


def verify_by_account(account_id: str, path: Path = LOCATION_DATA_PATH) -> dict[str, Any] | None:
    for record in load_location_records(path):
        if str(record.get("Acc.No", "")).strip() == str(account_id).strip():
            return verify_location(record)
    return None


def _split_wrapped_row(row: list[str]) -> list[str]:
    if len(row) == 1 and "," in row[0]:
        return next(csv.reader([row[0]]))
    return row


def _extract_evidence(record: dict[str, str], retrieved: list[dict[str, Any]]) -> list[Evidence]:
    evidence = []
    branch = record.get("Account Opening Branch Code (Last Six Characters of IFSC Code)", "")
    for doc in retrieved:
        if doc.get("kind") == "branch_code" and str(doc.get("code")) == str(branch):
            evidence.append(
                Evidence(
                    "branch_code",
                    doc.get("city", ""),
                    doc.get("state", ""),
                    2.0,
                    f"Branch code {branch} maps to {doc.get('city')}, {doc.get('state')}.",
                )
            )

    for source, field, weight in [
        ("address", "Address", 1.5),
        ("frequent_location", "Frequent Location", 1.2),
        ("last_location", "Last location", 1.0),
        ("upi_location", "UPI Location", 2.5),
    ]:
        location = record.get(field, "")
        state = _state_from_text(location)
        if state:
            evidence.append(Evidence(source, location, state, weight, f"{field} indicates {state}."))

    for source, field, weight in [
        ("driving_license", "DL Number", 2.0),
        ("vehicle_number", "Vehicle Number", 1.0),
    ]:
        code = _state_code_prefix(record.get(field, ""))
        state = STATE_BY_CODE.get(code, "")
        if state:
            evidence.append(Evidence(source, code, state, weight, f"{field} starts with {code}."))

    phone = record.get("Ph_no. ", "") or record.get("Ph_no.", "")
    if phone:
        phone_state = _phone_hint(phone)
        if phone_state:
            evidence.append(Evidence("phone_prefix", phone[:4], phone_state, 0.5, "Phone prefix heuristic."))

    return evidence


def _state_code_prefix(value: str) -> str:
    match = re.match(r"\s*([A-Z]{2})", str(value).upper())
    return match.group(1) if match else ""


def _state_from_text(value: str) -> str:
    text = str(value).lower()
    for state in STATE_BY_CODE.values():
        if state.lower() in text:
            return state
    for city, state in sorted(CITY_STATE_HINTS.items(), key=lambda item: len(item[0]), reverse=True):
        if city.lower() in text:
            return state
    return ""


def _phone_hint(value: str) -> str:
    first_digit = str(value).strip()[:1]
    return {
        "7": "Rajasthan",
        "8": "Maharashtra",
        "9": "Rajasthan",
        "6": "Madhya Pradesh",
    }.get(first_digit, "")


def _best_city_for_state(evidence: list[Evidence], state: str) -> str:
    cities = [item.location for item in evidence if item.state == state and len(item.location) > 2]
    return Counter(cities).most_common(1)[0][0] if cities else state
