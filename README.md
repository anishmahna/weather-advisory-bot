# Weather-Advisory Support Bot (SOP-grounded LangGraph agent)

Answers outdoor-activity safety questions from **live Open-Meteo data**, but only ever gives advice that
comes from a written SOP (`backend/sops/*.yaml`). No matching SOP -> "I don't have guidance for that".

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY (or set LLM_PROVIDER=openai + OPENAI_API_KEY)
uvicorn backend.main:app --port 8000          # backend  (POST /chat, GET /sops, GET /health)
streamlit run frontend/app.py                 # frontend (BACKEND_URL env var if not localhost:8000)
pytest evals -v                               # offline evals (no key/network needed)
RUN_LIVE=1 pytest evals -v -m live            # live evals (real Open-Meteo + real LLM)
```
Every reply has a "Why did it say that?" panel: route taken, matched SOP ids, primary SOP, facts shown, composer mode.

## Graph
```mermaid
flowchart TD
  S([user msg]) --> intake
  intake -->|parse/LLM error| failure
  intake -->|not outdoor| off_topic
  intake -->|no location known| need_location
  intake --> location
  location -->|unresolved / error| failure
  location --> weather
  weather -->|API error| failure
  weather --> match
  match -->|situational SOP| override --> composer
  match -->|regular SOP(s)| composer
  match -->|none| no_match
  composer & no_match & failure & off_topic & need_location --> remember --> E([reply])
```
Failure, no_match, off_topic and need_location return **fixed text with no LLM call**, so they cannot drift into a guess.

## SOPs: form and why
YAML files, validated by Pydantic at startup (bad file => app refuses to start and names the file).
*Why:* thresholds and advice text sit together in data a policy owner can edit without reading Python.
Conditions are a small tree (`all`/`any`/leaf `field op value`) over weather fields, plus `applies_to`
(activity tags/groups, audiences). 15 SOPs, 5 categories (situational, outdoor_exercise, travel,
vulnerable_groups, leisure), severities info->critical. Special kinds:
* **situational** (SIT-01/02): apply to any outdoor question, always lead. SIT-01 is the "rain system" rule:
  fires on heavy 24h rain **or** low pressure + (30 mm or 90% prob or heavy-rain code), so it triggers even when
  no single number is extreme. *Limitation:* Open-Meteo has no IMD low-pressure feed; this is a model-data proxy.
* **fuzzy** (FZ-01 picnic): no numeric trigger. Deterministic prefilter on intent, LLM yes/no on a soft
  criterion, id re-validated against the loaded set; advice is a qualitative rubric.
* **fallback** (EX-06, TR-03): only used if nothing else matched, so "calm day, cycling" gets an explicit
  "no threshold exceeded" instead of a shrug.

## Conflict rule (decided on purpose)
Situational SOPs lead. Everything else is ranked by severity, then `priority`, then id. Top one is the
primary; **all other matches are surfaced beneath it**. Why: silently dropping a real lower-ranked hazard is
worse than a slightly longer answer, and ranking is reproducible/explainable. (Code: `backend/matching.py`.)

## What is code vs model
| Deterministic code | LLM |
|---|---|
| geocoding, forecast, snapshot (`weather.py`) | intake: message -> structured intent (closed vocabulary) |
| all numeric/threshold matching, ranking (`matching.py`) | fuzzy SOP yes/no (ids re-validated) |
| fixed failure/no-match text | wording of the final reply |
| output validator (below) | |

**Enforcement of the non-negotiables** (`nodes/composer.py:validate_reply`): the composer sees only matched SOP text
+ fetched facts (never the raw user message); its reply must cite only loaded SOP ids, must cite the primary/lead
SOPs, and every number must exist in the fetched facts or SOP text. Otherwise a deterministic template is shipped
(`composer_mode: fallback`). Intake output is clamped to `config/taxonomy.yaml`.

## Adding an 11th SOP live
Add a block to any file in `backend/sops/` (or a new file). The registry hot-reloads on the next request; a broken
edit keeps the last good set and shows in `/health` and the trace. No Python touched (eval `test_16`).
Honest boundaries: a rule on a **new weather variable** needs one line in `config/weather_fields.yaml`
(still data); a new **activity word** needs a line in `config/taxonomy.yaml`; a genuinely new **kind of condition**
(e.g. comparing two fields) would need code.

## Eval results (what I actually ran)
Offline tier run here: **15 passed, 4 skipped** (the 4 are `live`). **I could not run the live tier in my
build sandbox (no Open-Meteo access, no LLM key), so those 4 are UNVERIFIED - run them before trusting this.**

| Test | Checks | Result |
|---|---|---|
| 00 | policy meets brief (>=10 SOPs, 3 categories, severities, fuzzy, situational) | pass |
| 01, 02 | SOP clearly applies; cited; real number in reply | pass |
| 03 | wording-independence of matcher | pass - **weak**: offline it only proves matching uses structured intent |
| 04 (live) | real-LLM paraphrase -> right intent | **not run** |
| 05 | severe case, no single extreme number, SIT-01 leads, API numbers cited | pass - **synthetic payload**, not a live capture |
| 06 (live) | live Bhopal vs independent oracle | **not run** |
| 07, 08 | no-SOP honest answer; baseline fallback SOP | pass |
| 09, 10 | API unreachable; location unresolved -> honest failure, no digits | pass |
| 11 | hostile LLM: fake SOP id + invented numbers get blocked | pass |
| 12 | injected user text never reaches the composer | pass |
| 13 (live) | real model + injection text | **not run** |
| 14 | conflict rule | pass |
| 15 | session follow-up inherits city/activity; new session doesn't | pass |
| 16, 17 | 11th SOP w/o code; malformed SOP refuses to start | pass |

### Known gaps / honesty notes
* **Live-data drift:** test 06 asserts against an oracle over whatever the API returns, so it keeps working after
  the MP system passes, but on a calm day it proves consistency only. Deterministic severe coverage comes from
  test 05's synthetic payload; better would be recording a real API payload during an event and replaying it.
* Test 05 uses a synthetic payload, so SIT-01's thresholds are not validated against real IMD events.
* Intake can mis-map an activity; the validator guards advice and numbers but not a wrong intent. Live test 04 probes this.
* Fuzzy matching is the one place the LLM influences *which* SOP applies; it is limited to a yes/no over
  prefiltered info-level candidates and fails closed.
* Geocoding takes the first candidate (documented default); ambiguity is not surfaced to the user.
* Session memory is in-process only (resets on restart, as the brief allows).
* Deployed URL and screen recording are not included - they need your hosting account (Streamlit Community
  Cloud / Render for the two services) and your voice.
