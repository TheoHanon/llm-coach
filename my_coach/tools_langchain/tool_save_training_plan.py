from langchain_core.tools import tool
from pydantic import BaseModel
from typing import List

from my_coach.domain.shemas import TrainingItem
from my_coach.config import settings

import json
from pathlib import Path
import pandas as pd


path = Path(settings.save_path)


class SaveArgs(BaseModel):
    training_plan: List[TrainingItem]


@tool("save-training-plan-tools", args_schema=SaveArgs)
def save_training_plan(training_plan: List[TrainingItem]) -> str:
    """
    Use this tool to save the training plan.
    """

    plan = [
        {"Date": item.Date, "Description": item.Description} for item in training_plan
    ]

    df = pd.DataFrame(plan)

    df = df.sort_values("Date").reset_index(drop=True)
    start, end = df["Date"].min(), df["Date"].max()
    df["Date"] = df["Date"].apply(lambda d: d.strftime("%d-%m-%Y"))
    df.to_csv(path, index=False)

    return json.dumps(
        {
            "status": "ok",
            "path": str(path),
            "rows_written": int(len(df)),
            "date_range": {
                "start": start.strftime("%d-%m-%Y"),
                "end": end.strftime("%d-%m-%Y"),
            },
        },
        ensure_ascii=False,
    )
