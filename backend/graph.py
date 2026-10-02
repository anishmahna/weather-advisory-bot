from __future__ import annotations
"""LangGraph agent. Control flow only; policy lives in sops/*.yaml, vocab in config/.

  intake -> {failure | off_topic | need_location | location}
  location -> {failure | weather};  weather -> {failure | match}
  match -> {override | composer | no_match}
  override -> composer;  every terminal -> remember -> END
"""
from typing import Optional, TypedDict
from langgraph.graph import StateGraph, START, END
from . import weather
from .config import Config, load_config
from .loader import SopRegistry
from .matching import rank_key
from .memory import SessionMemory
from .nodes.composer import build_payload, compose, render_facts
from .nodes.intake import IntakeError, run_intake
from .nodes.matcher import run_match

NO_SOP = " (No SOP applied, so no safety advice is given.)"
FIXED = {
    "location_unresolved": "I couldn't find that location, so I can't check the weather for it. Could you try a nearby city name?" + NO_SOP,
    "weather_unavailable": "I can't reach the live weather service right now, so I don't have a forecast and won't guess one. Please try again in a few minutes." + NO_SOP,
    "intake_failed": "Sorry, I couldn't understand that request well enough to check it against our policies. Could you rephrase it?" + NO_SOP,
    "off_topic": "I can only help with outdoor-activity safety and weather questions, and no SOP covers that, so I don't have guidance for it.",
    "need_location": "Happy to check - which city or place should I look at?" + NO_SOP,
}


class State(TypedDict, total=False):
    session_id: str
    message: str
    history: list
    facts: dict
    decisions: list
    intent: dict
    place: dict
    snapshot: dict
    error_kind: Optional[str]
    error_detail: Optional[str]
    match: dict
    reply: str
    route: str
    composer_mode: str
    composer_note: str
    shown_facts: list


def build_graph(registry: SopRegistry, cfg: Config, memory: SessionMemory):
    def intake(s: State):
        try:
            return {"intent": run_intake(s["message"], s["history"], s["facts"], cfg)}
        except IntakeError as e:
            return {"error_kind": "intake_failed", "error_detail": str(e)}

    def route_intake(s):
        if s.get("error_kind"):
            return "failure"
        if not s["intent"]["outdoor_related"]:
            return "off_topic"
        return "location" if s["intent"]["location"] else "need_location"

    def location(s: State):
        try:
            return {"place": weather.geocode(s["intent"]["location"])}
        except weather.WeatherError as e:
            return {"error_kind": e.kind, "error_detail": str(e)}

    def weather_node(s: State):
        try:
            raw = weather.fetch_forecast(s["place"]["lat"], s["place"]["lon"], cfg.fields)
            return {"snapshot": weather.build_snapshot(raw, s["intent"]["time_window"], cfg.taxonomy, cfg.fields)}
        except weather.WeatherError as e:
            return {"error_kind": e.kind, "error_detail": str(e)}

    ok_or_fail = lambda nxt: (lambda s: "failure" if s.get("error_kind") else nxt)

    def match(s: State):
        return {"match": run_match(registry.get(), s["intent"], s["snapshot"]["values"], cfg)}

    def route_match(s):
        m = s["match"]
        if m["override_ids"]:
            return "override"
        return "composer" if m["regular_ids"] else "no_match"

    def override(s: State):          # situational system leads; mark style for composer
        return {"route": "override"}

    def composer(s: State):
        by_id = {x.id: x for x in registry.get()}
        m, snap, intent = s["match"], s["snapshot"], s["intent"]
        lead = [by_id[i] for i in m["override_ids"]]
        regular = [by_id[i] for i in m["regular_ids"]]
        primary = lead[0] if lead else regular[0]
        # lead SOPs are required citations; additional = remaining regular matches in rank order
        additional = sorted([x for x in regular if x.id != primary.id], key=rank_key)
        referenced = set()
        for x in lead + regular:
            if x.conditions:
                referenced |= x.conditions.referenced_fields()
        place, when = s["place"]["display"], f"{snap['time']} local ({snap['window']})"
        facts = render_facts(snap["values"], snap["units"], cfg.fields["display"], referenced, place, when)
        previous = [{"place": d["place"], "when": d["when"], "sop_ids": d["sop_ids"]} for d in s["decisions"][-2:]]
        payload = build_payload(
            "lead_with_system" if lead else "standard",
            {"place": place, "when": when, "activity": intent["activity"], "audiences": intent["audiences"]},
            facts, m["missing_fields"], lead, primary, additional, previous)
        reply, mode, note = compose(payload)
        return {"reply": reply, "composer_mode": mode, "composer_note": note, "shown_facts": facts,
                "route": s.get("route") or "compose"}

    def no_match(s: State):
        return {"route": "no_match", "reply": (
            "I don't have guidance for that. None of our written safety policies (SOPs) apply to this "
            "activity under the conditions I fetched, and I'd rather say so than guess. "
            "That is not a statement that it is safe." + NO_SOP)}

    fixed = lambda key, route: (lambda s: {"route": route, "reply": FIXED[key]})
    def failure(s: State):
        return {"route": "failure", "reply": FIXED[s["error_kind"]]}

    def remember(s: State):
        intent = s.get("intent") or {}
        m = s.get("match") or {}
        sop_ids = m.get("override_ids", []) + m.get("regular_ids", [])
        decision = None
        if s.get("route") in ("override", "compose", "no_match"):
            decision = {"place": s["place"]["display"], "when": s["snapshot"]["window"], "sop_ids": sop_ids}
        facts = None
        if intent.get("outdoor_related"):
            facts = {k: intent.get(k) for k in ("location", "activity", "audiences", "time_window")}
        memory.update(s["session_id"], s["message"], s["reply"], facts, decision)
        return {}

    g = StateGraph(State)
    for name, fn in [("intake", intake), ("location", location), ("weather", weather_node), ("match", match),
                     ("override", override), ("composer", composer), ("no_match", no_match),
                     ("failure", failure), ("off_topic", fixed("off_topic", "off_topic")),
                     ("need_location", fixed("need_location", "need_location")), ("remember", remember)]:
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", route_intake, {k: k for k in ["failure", "off_topic", "need_location", "location"]})
    g.add_conditional_edges("location", ok_or_fail("weather"), {"failure": "failure", "weather": "weather"})
    g.add_conditional_edges("weather", ok_or_fail("match"), {"failure": "failure", "match": "match"})
    g.add_conditional_edges("match", route_match, {"override": "override", "composer": "composer", "no_match": "no_match"})
    g.add_edge("override", "composer")
    for t in ["composer", "no_match", "failure", "off_topic", "need_location"]:
        g.add_edge(t, "remember")
    g.add_edge("remember", END)
    return g.compile()


class Advisor:
    def __init__(self, registry: SopRegistry | None = None, memory: SessionMemory | None = None, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.registry = registry or SopRegistry(self.cfg)
        self.memory = memory or SessionMemory()
        self.graph = build_graph(self.registry, self.cfg, self.memory)

    def chat(self, session_id: str, message: str) -> dict:
        mem = self.memory.get(session_id)
        out = self.graph.invoke({"session_id": session_id, "message": message, "history": mem["history"],
                                 "facts": mem["facts"], "decisions": mem["decisions"]})
        m = out.get("match") or {}
        trace = {
            "route": out.get("route"), "intent": out.get("intent"),
            "place": (out.get("place") or {}).get("display"),
            "override_sop_ids": m.get("override_ids", []), "matched_sop_ids": m.get("regular_ids", []),
            "primary_sop_id": m.get("primary_id"), "used_fallback_sop": m.get("used_fallback", False),
            "facts_shown": out.get("shown_facts"), "composer_mode": out.get("composer_mode"),
            "composer_note": out.get("composer_note"), "error": out.get("error_detail"),
            "sop_load_error": self.registry.last_error,
        }
        return {"reply": out["reply"], "trace": trace}
