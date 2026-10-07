"""Small source-reported measurement fields; unknown conditions stay null."""

from pydantic import BaseModel, ConfigDict, Field


class ExperimentalConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = Field(default=None, max_length=200)
    hardware: str | None = Field(default=None, max_length=300)
    software: str | None = Field(default=None, max_length=200)
    workload: str | None = Field(default=None, max_length=400)
    slo: str | None = Field(default=None, max_length=200)


class MeasurementFields(BaseModel):
    metric: str | None = Field(default=None, max_length=100)
    value: str | None = Field(default=None, max_length=100)
    unit: str | None = Field(default=None, max_length=40)
    baseline: str | None = Field(default=None, max_length=300)
    conditions: ExperimentalConditions = Field(default_factory=ExperimentalConditions)


def measurement_fields(finding):
    data = {
        key: getattr(finding, key) for key in ("metric", "value", "unit", "baseline")
    }
    if not any(data.values()):
        return {}
    return {**data, "conditions": finding.conditions.model_dump()}
