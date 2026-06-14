from dataclasses import asdict, dataclass
from typing import Any

from .rag import Evidence, retrieve_knowledge, _extract_evidence


@dataclass
class AgentResult:
    agent: str
    summary: str
    score: float
    evidence: list[dict[str, Any]]
    flags: list[str]


def run_location_agents(record: dict[str, str]) -> dict[str, Any]:
    """Run the local 4-agent RAG pipeline for Task 2.

    This is intentionally local and deterministic: the "agents" are specialized
    reasoning stages over retrieved evidence, not paid LLM calls.
    """
    planner = planner_agent(record)
    static = static_identity_agent(record, planner["retrieved_context"])
    activity = activity_location_agent(record, planner["retrieved_context"])
    final = conflict_resolver_agent(record, static, activity, planner)

    return {
        "account_id": record.get("Acc.No", ""),
        "pipeline": "local_multi_agent_rag",
        "planner": planner,
        "agents": [asdict(static), asdict(activity), asdict(final)],
        "final": {
            "predicted_location": final.summary,
            "confidence": _confidence(final.score),
            "score": round(final.score, 2),
            "manual_review": "manual_review" in final.flags,
            "flags": final.flags,
        },
    }


def planner_agent(record: dict[str, str]) -> dict[str, Any]:
    query_parts = []
    for field in [
        "Address",
        "Account Opening Branch Code (Last Six Characters of IFSC Code)",
        "DL Number",
        "Vehicle Number",
        "Frequent Location",
        "Last location",
        "UPI Location",
        "ATM Transiction",
    ]:
        value = record.get(field, "")
        if value:
            query_parts.append(f"{field}: {value}")

    query = " | ".join(query_parts)
    return {
        "agent": "Planner & Query Builder",
        "static_checks": ["branch_code", "address", "dl_state", "vehicle_state", "phone_prefix"],
        "activity_checks": ["upi_location", "frequent_location", "last_location", "atm_transaction"],
        "retrieval_query": query,
        "retrieved_context": retrieve_knowledge(query, top_k=8),
    }


def static_identity_agent(record: dict[str, str], retrieved_context: list[dict[str, Any]]) -> AgentResult:
    evidence = [
        item
        for item in _extract_evidence(record, retrieved_context)
        if item.source in {"branch_code", "address", "driving_license", "vehicle_number", "phone_prefix"}
    ]
    flags = _conflict_flags(evidence)
    score = sum(item.weight for item in evidence) - len(flags)
    summary = _top_location_summary(evidence) or "Static identity location unknown"
    return AgentResult(
        agent="Static Identity Verifier",
        summary=summary,
        score=max(float(score), 0.0),
        evidence=[asdict(item) for item in evidence],
        flags=flags,
    )


def activity_location_agent(record: dict[str, str], retrieved_context: list[dict[str, Any]]) -> AgentResult:
    evidence = [
        item
        for item in _extract_evidence(record, retrieved_context)
        if item.source in {"upi_location", "frequent_location", "last_location"}
    ]
    atm_value = record.get("ATM Transiction", "")
    flags = _conflict_flags(evidence)
    if atm_value and atm_value.lower() not in {"na", "no", ""}:
        flags.append("atm_activity_present")
    score = sum(item.weight for item in evidence) - max(len(flags) - 1, 0)
    summary = _top_location_summary(evidence) or "Activity location unknown"
    return AgentResult(
        agent="Activity Location Verifier",
        summary=summary,
        score=max(float(score), 0.0),
        evidence=[asdict(item) for item in evidence],
        flags=flags,
    )


def conflict_resolver_agent(
    record: dict[str, str],
    static: AgentResult,
    activity: AgentResult,
    planner: dict[str, Any],
) -> AgentResult:
    evidence = [Evidence(**item) for item in static.evidence + activity.evidence]
    states = {item.state for item in evidence if item.state}
    flags = []
    if len(states) >= 3:
        flags.append("multi_state_conflict")
    if static.summary != activity.summary and "unknown" not in static.summary.lower() and "unknown" not in activity.summary.lower():
        flags.append("static_activity_mismatch")

    score = static.score + activity.score - (1.5 * len(flags))
    if score < 4 or flags:
        flags.append("manual_review")

    summary = _top_location_summary(evidence) or "Location inconclusive"
    return AgentResult(
        agent="Conflict Resolver & Final Scorer",
        summary=summary,
        score=max(float(score), 0.0),
        evidence=[asdict(item) for item in evidence],
        flags=sorted(set(flags)),
    )


def _top_location_summary(evidence: list[Evidence]) -> str:
    if not evidence:
        return ""
    scores: dict[tuple[str, str], float] = {}
    for item in evidence:
        key = (item.location or item.state, item.state)
        scores[key] = scores.get(key, 0.0) + item.weight
    location, state = max(scores.items(), key=lambda item: item[1])[0]
    return f"{location}, {state}" if location and state and location != state else state or location


def _conflict_flags(evidence: list[Evidence]) -> list[str]:
    states = {item.state for item in evidence if item.state}
    return ["state_conflict"] if len(states) > 1 else []


def _confidence(score: float) -> str:
    if score >= 7:
        return "High"
    if score >= 4:
        return "Medium"
    return "Low"
