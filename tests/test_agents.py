from ficoforce.agents import run_location_agents


def test_run_location_agents_returns_four_stage_result():
    record = {
        "Acc.No": "123456",
        "Address": "GandhiNagar Jaipur Rajasthan",
        "Ph_no. ": "9033068120",
        "Account Opening Branch Code (Last Six Characters of IFSC Code)": "263148",
        "DL Number": "MP36202457293",
        "Vehicle Number": "MP41ZP3891",
        "Frequent Location": "Guwahati",
        "Last location": "Ahmedabad",
        "UPI Location": "Jaipur",
        "ATM Transiction": "12-123-1234",
    }

    result = run_location_agents(record)

    assert result["pipeline"] == "local_multi_agent_rag"
    assert result["planner"]["retrieved_context"]
    assert len(result["agents"]) == 3
    assert result["final"]["confidence"] in {"High", "Medium", "Low"}
