from typing import Literal
from pydantic import BaseModel, Field, field_validator
from datetime import date, datetime


class WelcomeRoute(BaseModel):
    mode: Literal["make", "discuss"] = Field(
        description="The chosen mode. 'make' = create a new plan, 'discuss' = talk about an existing one."
    )


class ModifyRoute(BaseModel):
    mode: Literal["modify", "continue"] = Field(
        description="The chosen mode. 'modify' = modify the current plan, 'continue' = don't want to modify."
    )


class TrainingItem(BaseModel):
    Date: date = Field(
        description="Session date, format DD-MM-YYYY.", examples=["12-08-2025"]
    )
    Description: str = Field(
        description="Free-text description of the session.",
        examples=["10 km easy + strides"],
    )

    @field_validator("Date", mode="before")
    @classmethod
    def parse_date(cls, value):
        if isinstance(value, str):
            s = value.strip()
            for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt).date()
                except ValueError:
                    pass
            raise ValueError("Date must be DD-MM-YYYY or YYYY-MM-DD.")
        raise TypeError("Date must be a string or a date.")


class TrainingPlan(BaseModel):
    plan: list[TrainingItem]
    justification: str
