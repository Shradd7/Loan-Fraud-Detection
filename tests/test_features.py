import pandas as pd

from ficoforce.features import prepare_credit_features


def test_prepare_credit_features_adds_risk_signals():
    frame = pd.DataFrame(
        [
            {
                "SI_FLG": "Y",
                "LOCKER_HLDR_IND": "N",
                "UID_FLG": "Y",
                "KYC_FLG": "Y",
                "INB_FLG": "N",
                "EKYC_FLG": "Y",
                "ALL_LON_LIMIT": 1200,
                "INCOME_BAND1": "G",
                "AVERAGE_ACCT_AGE1": "2yrs 9mon",
                **{f"{word}MNTHSDR": 150 for word in ["ONE", "TWO", "THREE"]},
            }
        ]
    )

    result = prepare_credit_features(frame)

    assert result.loc[0, "SI_FLG"] == 1
    assert result.loc[0, "INCOME_BAND1"] == 7
    assert result.loc[0, "AVERAGE_ACCT_AGE1"] == 33
    assert "overspend_ratio" in result.columns
    assert result.loc[0, "max_consec_overspend"] == 3
