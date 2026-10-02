"""Compose node: the LLM sees ONLY matched SOP text + fetched facts (never the raw user
message). Its output is then machine-checked; on any violation we ship a deterministic
template instead. This is where "numbers come from the API" and "no invented SOPs"
are enforced in code."""
import json, re
from .. import llm

SYSTEM = """[TASK:compose]
You write the final reply to a user from the JSON provided. Use ONLY the facts and SOP texts in it.
Rules:
1. Do not add safety advice beyond the SOP text; rephrase, never extend.
2. Quote numbers exactly as given in facts. Never round, convert, estimate or add numbers. Do not use numbered lists.
3. Cite every SOP you use like [SOP-XX-00]. Always cite primary_sop and every lead_sops entry.
4. If compose_style is "lead_with_system": begin with the lead_sops (the rain/storm system) and its figures, before activity advice.
5. Mention each additional_sops entry briefly after the primary, in the order given. Say the primary was chosen as the highest-severity applicable SOP.
6. If unavailable_data is non-empty, say those readings were unavailable.
7. Never describe an activity as safe or guaranteed. Be kind, plain and concise (under 150 words).
8. previous_turns are earlier decisions in this chat: stay consistent with them or explain what changed (time/place)."""

_NUM = re.compile(r"\d+(?:\.\d+)?")
_ID = re.compile(r"SOP-[A-Z]+-\d+")


def render_facts(values: dict, units: dict, display: list, referenced: set, place: str, when: str) -> list[str]:
    names = list(dict.fromkeys(list(display) + sorted(referenced)))
    out = [f"location: {place}", f"time: {when}"]
    for n in names:
        if n == "local_hour" or values.get(n) is None:
            continue
        u = units.get(n, "")
        out.append(f"{n}: {values[n]}{(' ' + u) if u and u != 'wmo code' else ''}")
    return out


def _sop_dict(s):
    return {"id": s.id, "severity": s.severity, "title": s.title, "advice": " ".join(s.advice.split()), "cite_as": s.cite_as}


def build_payload(style, context, facts, unavailable, lead, primary, additional, previous):
    return {"compose_style": style, "context": context, "facts": facts, "unavailable_data": unavailable,
            "primary_sop": _sop_dict(primary), "lead_sops": [_sop_dict(s) for s in lead],
            "additional_sops": [_sop_dict(s) for s in additional], "previous_turns": previous}


def validate_reply(reply: str, payload: dict) -> tuple[bool, str]:
    sops = [payload["primary_sop"]] + payload["lead_sops"] + payload["additional_sops"]
    allowed_ids = {s["id"] for s in sops}
    cited = set(_ID.findall(reply))
    if not cited <= allowed_ids:
        return False, f"cited unknown SOP ids {sorted(cited - allowed_ids)}"
    required = {payload["primary_sop"]["id"]} | {s["id"] for s in payload["lead_sops"]}
    if not required <= cited:
        return False, f"missing required citations {sorted(required - cited)}"
    allowed_text = " ".join(payload["facts"] + [s["advice"] + " " + s["title"] + " " + s["cite_as"] for s in sops])
    allowed_text = _ID.sub("", allowed_text)
    allowed_nums = {float(x) for x in _NUM.findall(allowed_text)}
    extra = {x for x in _NUM.findall(_ID.sub("", reply)) if float(x) not in allowed_nums}
    if extra:
        return False, f"numbers not present in fetched data/SOP text: {sorted(extra)}"
    return True, "ok"


def deterministic_reply(payload: dict) -> str:
    lines = [f"Here is what the live forecast shows ({payload['context']['place']}, {payload['context']['when']}):"]
    lines += [f"- {f}" for f in payload["facts"][2:]]
    if payload["unavailable_data"]:
        lines.append(f"Some readings were unavailable: {', '.join(payload['unavailable_data'])}.")
    for s in payload["lead_sops"]:
        lines.append(f"\n[{s['id']}] ({s['severity']}) {s['title']}: {s['advice']}")
    p = payload["primary_sop"]
    if p["id"] not in {s["id"] for s in payload["lead_sops"]}:
        lines.append(f"\nPrimary policy (highest severity applicable) [{p['id']}] ({p['severity']}) {p['title']}: {p['advice']}")
    for s in payload["additional_sops"]:
        lines.append(f"\nAlso applies [{s['id']}] ({s['severity']}) {s['title']}: {s['advice']}")
    return "\n".join(lines)


def compose(payload: dict) -> tuple[str, str, str]:
    """returns (reply, mode, note)"""
    try:
        reply = llm.complete(SYSTEM, json.dumps(payload)).strip()
        ok, why = validate_reply(reply, payload)
        if ok:
            return reply, "llm", "ok"
        return deterministic_reply(payload), "fallback", why
    except Exception as e:
        return deterministic_reply(payload), "fallback", f"llm error: {e}"
