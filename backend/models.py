"""Pydantic schemas. SOP files are validated against these at startup."""
from __future__ import annotations
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator

Severity = Literal["info", "low", "moderate", "high", "critical"]
SEVERITY_RANK = {"info": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}
Op = Literal["gt", "gte", "lt", "lte", "eq", "in"]


class Condition(BaseModel):
    """Either a leaf (field/op/value) or a nested all:[...] / any:[...] group."""
    model_config = {"extra": "forbid"}
    field: Optional[str] = None
    op: Optional[Op] = None
    value: Any = None
    all: Optional[list["Condition"]] = None
    any: Optional[list["Condition"]] = None

    @model_validator(mode="after")
    def _shape(self):
        forms = sum([self.field is not None, self.all is not None, self.any is not None])
        if forms != 1:
            raise ValueError("condition must be exactly one of: leaf (field/op/value), all, any")
        if self.field is not None and (self.op is None or self.value is None):
            raise ValueError(f"leaf condition on '{self.field}' needs op and value")
        for grp in (self.all, self.any):
            if grp is not None and not grp:
                raise ValueError("all/any groups must not be empty")
        return self

    def referenced_fields(self) -> set[str]:
        if self.field is not None:
            return {self.field}
        out: set[str] = set()
        for child in (self.all or self.any or []):
            out |= child.referenced_fields()
        return out


Condition.model_rebuild()


class AppliesTo(BaseModel):
    model_config = {"extra": "forbid"}
    activities: list[str] = []
    activity_groups: list[str] = []
    audiences: list[str] = []


class SOP(BaseModel):
    model_config = {"extra": "forbid"}
    id: str = Field(pattern=r"^SOP-[A-Z]+-\d+$")
    category: str
    severity: Severity
    title: str
    advice: str
    cite_as: str
    situational: bool = False        # outranks everything, applies to any outdoor question
    fallback: bool = False           # only used if nothing else matched
    priority: int = 0                # tie-break within equal severity (higher wins)
    applies_to: AppliesTo = AppliesTo()
    conditions: Optional[Condition] = None
    fuzzy_criterion: Optional[str] = None   # soft, LLM-judged criterion

    @model_validator(mode="after")
    def _needs_trigger(self):
        if not (self.conditions or self.fuzzy_criterion or self.fallback):
            raise ValueError("SOP needs conditions, a fuzzy_criterion, or fallback: true")
        if self.fallback and self.situational:
            raise ValueError("an SOP cannot be both fallback and situational")
        return self


class Intent(BaseModel):
    outdoor_related: bool
    location: Optional[str] = None
    activity: Optional[str] = None
    audiences: list[str] = []
    time_window: Optional[str] = None
    intent_summary: str = ""
