"""Pure, deterministic SOP matching. No LLM, no I/O.

CONFLICT RULE (decided on purpose):
  1. Situational SOPs (e.g. active rain system) ALWAYS lead and are named first.
  2. All other matched SOPs are ranked by severity, then `priority`, then id.
  3. The top-ranked SOP is the PRIMARY; every other match is surfaced beneath it.
  Why: dropping a lower-ranked but real hazard silently is worse than a longer
  answer; ranking makes the order explainable and reproducible.
"""
from typing import Callable, Optional
from .config import Config
from .models import SOP, Condition, SEVERITY_RANK

_OPS = {
    "gt": lambda a, b: a > b, "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b, "lte": lambda a, b: a <= b,
    "eq": lambda a, b: a == b, "in": lambda a, b: a in b,
}


def eval_condition(node: Condition, values: dict) -> bool:
    if node.all is not None:
        return all(eval_condition(n, values) for n in node.all)
    if node.any is not None:
        return any(eval_condition(n, values) for n in node.any)
    v = values.get(node.field)
    if v is None:
        return False
    try:
        return bool(_OPS[node.op](v, node.value))
    except TypeError:
        return False


def intent_applies(sop: SOP, intent: dict, cfg: Config) -> bool:
    a = intent.get("activity") or "other"
    ap = sop.applies_to
    if ap.activities or ap.activity_groups:
        if not (a in ap.activities or cfg.groups_for(a) & set(ap.activity_groups)):
            return False
    if ap.audiences and not (set(ap.audiences) & set(intent.get("audiences") or [])):
        return False
    return True


def rank_key(s: SOP):
    return (-SEVERITY_RANK[s.severity], -s.priority, s.id)


def match_sops(sops: list[SOP], intent: dict, values: dict, cfg: Config,
               fuzzy_fn: Optional[Callable[[str, list[SOP]], list[str]]] = None) -> dict:
    applicable = [s for s in sops if intent_applies(s, intent, cfg)]
    missing: set[str] = set()

    def cond_ok(s: SOP) -> bool:
        if s.conditions is None:
            return True
        ok = eval_condition(s.conditions, values)
        if not ok:
            missing.update(f for f in s.conditions.referenced_fields() if values.get(f) is None)
        return ok

    matched = [s for s in applicable if not s.fallback and not s.fuzzy_criterion and s.conditions and cond_ok(s)]
    fuzzy_cands = [s for s in applicable if not s.fallback and s.fuzzy_criterion and cond_ok(s)]
    if fuzzy_cands and fuzzy_fn:
        valid = {s.id for s in fuzzy_cands}
        picked = {i for i in fuzzy_fn(intent.get("intent_summary", ""), fuzzy_cands) if i in valid}  # invented ids dropped
        matched += [s for s in fuzzy_cands if s.id in picked]
    used_fallback = False
    if not matched:
        matched = [s for s in applicable if s.fallback and cond_ok(s)]
        used_fallback = bool(matched)
    override = sorted([s for s in matched if s.situational], key=rank_key)
    regular = sorted([s for s in matched if not s.situational], key=rank_key)
    primary = (override or regular or [None])[0]
    return {
        "override_ids": [s.id for s in override],
        "regular_ids": [s.id for s in regular],
        "primary_id": primary.id if primary else None,
        "used_fallback": used_fallback,
        "missing_fields": sorted(missing),
    }
