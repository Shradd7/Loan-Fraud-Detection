import re
from typing import Iterable

import numpy as np
import pandas as pd


MONTH_WORDS = [
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
]


def convert_duration_to_months(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()
    years = re.search(r"(\d+)\s*yrs?", text)
    months = re.search(r"(\d+)\s*(?:months?|mon)", text)
    return (int(years.group(1)) * 12 if years else 0) + (
        int(months.group(1)) if months else 0
    )


def _numeric_frame(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    present = [column for column in columns if column in df.columns]
    if not present:
        return pd.DataFrame(index=df.index)
    return df[present].apply(pd.to_numeric, errors="coerce").fillna(0)


def _row_slope(values: np.ndarray) -> np.ndarray:
    x = np.arange(values.shape[1])
    slopes = []
    for row in values:
        if np.all(np.isnan(row)) or np.nanstd(row) == 0:
            slopes.append(0.0)
        else:
            slopes.append(float(np.polyfit(x, np.nan_to_num(row), 1)[0]))
    return np.array(slopes)


def prepare_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature engineering used by training and serving."""
    out = df.copy()

    flag_columns = ["SI_FLG", "LOCKER_HLDR_IND", "UID_FLG", "KYC_FLG", "INB_FLG", "EKYC_FLG"]
    for column in flag_columns:
        if column in out.columns:
            out[column] = out[column].replace({"Y": 1, "N": 0}).astype(float)

    duration_columns = [
        "AVERAGE_ACCT_AGE1",
        "CREDIT_HISTORY_LENGTH1",
        "OLDEST_LON_TAKEN",
        "LATEST_LON_TAKEN",
        "LATEST_RESIDUAL_TENURE",
        "OLDEST_RESIDUAL_TENURE",
    ]
    for column in duration_columns:
        if column in out.columns and out[column].dtype == object:
            out[column] = out[column].map(convert_duration_to_months)

    income_band_mapping = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
        "M": 13,
        "EX05": 14,
    }
    if "INCOME_BAND1" in out.columns:
        out["INCOME_BAND1"] = out["INCOME_BAND1"].map(
            lambda value: income_band_mapping.get(str(value).strip(), np.nan)
        )

    if "ONEMNTHCR" in out.columns and "ONEMNTHSCR" not in out.columns:
        out.rename(columns={"ONEMNTHCR": "ONEMNTHSCR"}, inplace=True)

    sdr_columns = [f"{word}MNTHSDR" for word in MONTH_WORDS]
    sdr = _numeric_frame(out, sdr_columns)
    if not sdr.empty:
        limit = pd.to_numeric(out.get("ALL_LON_LIMIT", 0), errors="coerce").fillna(0) / 12
        total_spend = sdr.sum(axis=1)
        total_overspend = sdr.sub(limit, axis=0).clip(lower=0).sum(axis=1)
        out["overspend_ratio"] = np.where(total_spend > 0, total_overspend / total_spend, 0)
        overspend_flags = sdr.gt(limit, axis=0).to_numpy()
        out["max_consec_overspend"] = [
            _max_consecutive_true(row) for row in overspend_flags
        ]

    out_columns = [f"{word}MNTHOUTSTANGBAL" for word in reversed(MONTH_WORDS)]
    out_bal = _numeric_frame(out, out_columns)
    if not out_bal.empty:
        out["outbal_slope"] = _row_slope(out_bal.to_numpy())
        out["outbal_is_declining"] = (out["outbal_slope"] < 0).astype(int)

    debit_columns = [f"{word}MNTHAVGMTD" for word in MONTH_WORDS]
    debit = _numeric_frame(out, debit_columns)
    if not debit.empty:
        out["slope_MTD"] = _row_slope(debit.to_numpy())

    if "TIME_PERIOD" in out.columns:
        out["TIME_PERIOD_NUM"] = out["TIME_PERIOD"].map(_time_period_to_number)
        out.drop(columns=["TIME_PERIOD"], inplace=True)

    return out


def _max_consecutive_true(values) -> int:
    best = 0
    current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _time_period_to_number(value):
    if pd.isna(value):
        return np.nan
    text = str(value).upper().strip()
    match = re.match(r"([A-Z]{3})(\d{2})", text)
    if not match:
        return np.nan
    months = {
        "JAN": 1,
        "FEB": 2,
        "MAR": 3,
        "APR": 4,
        "MAY": 5,
        "JUN": 6,
        "JUL": 7,
        "AUG": 8,
        "SEP": 9,
        "OCT": 10,
        "NOV": 11,
        "DEC": 12,
    }
    month = months.get(match.group(1))
    year = int(match.group(2)) + 2000
    return year * 12 + month if month else np.nan
