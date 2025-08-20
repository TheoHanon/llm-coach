import json
from pathlib import Path
from datetime import datetime, date

from langchain_core.tools import tool

import pandas as pd
from my_coach.config import settings


path = Path(settings.save_path)


def _parse_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%d-%m-%Y").date()


@tool("load_training_plan", return_direct=False)
def load_training_plan() -> str:
    """
    Use this to load the training plan
    """

    if not path.exists():
        return json.dumps(
            {"error": "The file doesn't exist.", "path": str(path)}, ensure_ascii=False
        )

    df = pd.read_csv(path, dtype={"Date": str, "Description": str})

    expected = {"Date", "Description"}
    if not expected.issubset(df.columns):
        return json.dumps(
            {
                "status": "failed",
                "error": f"CSV must contain columns {sorted(expected)}.",
                "path": str(path),
            },
            ensure_ascii=False,
        )

    # Normalize/validate dates and sort
    try:
        df["Date"] = df["Date"].apply(_parse_date)
    except Exception:
        return json.dumps(
            {"status": "failed", "error": 'Dates must be in "DD-MM-YYYY" format.'},
            ensure_ascii=False,
        )

    df = df.sort_values("Date").reset_index(drop=True)

    # Convert back to string for JSON
    df["Date"] = df["Date"].apply(lambda d: d.strftime("%d-%m-%Y"))
    records = df[["Date", "Description"]].to_dict(orient="records")

    out = {"status": "ok", "plan": records}
    return json.dumps(out)
