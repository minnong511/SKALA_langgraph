"""보고서 평가 결과의 공통 형식."""

from typing import Literal, TypedDict


class ReportQualityResult(TypedDict):
    passed: bool
    groundedness: bool
    neutrality: bool
    bias_control: bool
    perspective_coverage: bool
    feedback: list[str]
    status: Literal["passed", "revise", "needs_review"]
