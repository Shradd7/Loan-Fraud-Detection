from ficoforce.rag import verify_location


def test_verify_location_returns_evidence_and_confidence():
    record = {
        "Acc.No": "123456",
        "First_Name_Middle_Name": "Shraddhan",
        "Last_Name ": "Singhai",
        "Ph_no. ": "9033068120",
        "Address": "GandhiNagar Jaipur Rajasthan",
        "Account Opening Branch Code (Last Six Characters of IFSC Code)": "263148",
        "DL Number": "MP36202457293",
        "Vehicle Number": "MP41ZP3891",
        "Frequent Location": "Guwahati",
        "Last location": "Ahmedabad",
        "UPI Location": "Jaipur",
    }

    result = verify_location(record)

    assert result["account_id"] == "123456"
    assert result["predicted_state"] in {"Rajasthan", "Madhya Pradesh"}
    assert result["confidence"] in {"High", "Medium", "Low"}
    assert result["evidence"]
