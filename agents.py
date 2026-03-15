"""
agents.py

All Groq LLM agent logic.

Three specialised agents, each with a narrow responsibility:

  OrchestratorAgent  - reads user intent + conversation history, decides
                       routing (edit_planner | clarifier | undo | reset).

  EditPlannerAgent   - converts a clear natural language request into an
                       ordered list of tool calls with parameters.

  ClarifierAgent     - when intent is ambiguous, generates a clarifying
                       question with 2-4 concrete options, each pre-planned
                       as tool steps so no extra LLM call is needed on click.

run_agentic_pipeline() wires the three together into a single call the UI uses.

All agents communicate via JSON. _groq_call() is the single integration point -
swap the client here to use any OpenAI-compatible LLM provider.
"""

import json
import streamlit as st
from groq import Groq
from tools import build_tool_list_str


# client initialization with caching so it's created once per session, not on every call
@st.cache_resource
def get_groq_client() -> Groq | None:

    key = st.secrets.get("GROQ_API_KEY", "")
    if not key:
        return None
    return Groq(api_key=key)


# system prompts for each agent

_ORCHESTRATOR_SYSTEM = """You are the orchestrator of an AI image editing pipeline.

Read the user's message and conversation history, then return ONE of:

{"action":"route","target":"edit_planner"}   — intent is clear enough to execute
{"action":"route","target":"clarifier"}      — intent is ambiguous (e.g. "fix it", "make it better", "make it pop")
{"action":"undo"}                            — user wants to undo / go back / revert
{"action":"reset"}                           — user wants to reset / start over / see original
{"action":"no_image","message":"..."}        — no image is loaded, or request is off-topic

Respond with raw JSON only. No markdown, no explanation outside the JSON."""


_EDIT_PLANNER_SYSTEM = """You are an expert image editing planner.

Available tools:
{tools}

Convert the user's natural language request into an ordered list of tool calls.
Respond with raw JSON only:

{{"steps":[{{"tool":"<name>","params":{{...}}}}, ...],"message":"Brief friendly description"}}

Key mappings (use these as defaults):
- vintage / retro              → apply_vintage
- cinematic / film look        → apply_cinematic
- hdr / vivid / pop            → apply_hdr
- brighter / lighten           → adjust_brightness value:35
- darker / dim                 → adjust_brightness value:-35
- much darker                  → adjust_brightness value:-65
- more contrast                → adjust_contrast value:50
- warmer / warm tones          → apply_warmth value:-40
- cooler / cool tones          → apply_warmth value:40
- sharpen / crisp              → apply_sharpen_filter
- blur / soften / dreamy       → apply_blur radius:8
- black and white / b&w / mono → apply_grayscale
- sepia                        → apply_sepia intensity:1.0
- cartoon                      → apply_cartoon
- vignette                     → apply_vignette strength:0.5
- remove background / cutout   → remove_background
- detect / find faces          → detect_faces
- rotate left                  → rotate_image angle:90
- rotate right                 → rotate_image angle:-90
- flip / mirror                → flip_horizontal
- undo / reset                 → reset

Chain naturally: "vintage but brighter" = apply_vintage + adjust_brightness value:25
Keep params within their stated ranges."""


_CLARIFIER_SYSTEM = """You are an image editing clarifier. The user's request is ambiguous.

Available tools:
{tools}

Generate a short clarifying question with 2-4 concrete options.
Each option must include pre-planned tool steps so no extra LLM call is needed.

Respond with raw JSON only:

{{"question":"Short clarifying question?",
  "options":["Option A","Option B","Option C"],
  "option_steps":[
    [{{"tool":"...","params":{{...}}}}],
    [{{"tool":"...","params":{{...}}}}],
    [{{"tool":"...","params":{{...}}}}]
  ],
  "message":"Short note to user"}}"""


# shared llm call

def _groq_call(system: str, messages: list[dict], max_tokens: int = 600) -> dict:

    client = get_groq_client()
    if not client:
        return {
            "action": "error",
            "message": (
                "No GROQ_API_KEY found. "
                "Add it to .streamlit/secrets.toml locally, "
                "or to Streamlit Cloud -> Settings -> Secrets when deployed."
            ),
        }
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.15,   # low temp = deterministic, consistent tool calls
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if the model wraps output in them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except json.JSONDecodeError as e:
        return {"action": "error", "message": f"Model returned malformed JSON: {e}"}
    except Exception as e:
        return {"action": "error", "message": f"Groq API error: {e}"}


# conversation history

def _build_history() -> list[dict]:

 
    # Sending history lets the agents understand follow-up requests like 'make it a bit more', 'undo that', 'now try cinematic instead'.
    # 8 turns covers all practical multi-step editing sessions.
    
    out = []
    for msg in st.session_state.chat_messages[-8:]:
        role = "user" if msg["role"] == "user" else "assistant"
        out.append({"role": role, "content": msg["content"]})
    return out


# agents

def orchestrator_agent(user_message: str) -> dict:

    # OrchestratorAgent: classify intent and return a routing decision, output is always a tiny JSON object.
    
    history = _build_history()
    return _groq_call(
        _ORCHESTRATOR_SYSTEM,
        history + [{"role": "user", "content": user_message}],
        max_tokens=120,
    )


def edit_planner_agent(user_message: str) -> dict:

    # EditPlannerAgent: build an ordered list of tool calls from a clear request

    system = _EDIT_PLANNER_SYSTEM.format(tools=build_tool_list_str())
    history = _build_history()
    return _groq_call(
        system,
        history + [{"role": "user", "content": user_message}],
        max_tokens=600,
    )


def clarifier_agent(user_message: str) -> dict:

    # ClarifierAgent: generate a clarifying question + pre-planned option steps
    
    system = _CLARIFIER_SYSTEM.format(tools=build_tool_list_str())
    return _groq_call(
        system,
        [{"role": "user", "content": user_message}],
        max_tokens=500,
    )


# main pipeline

def run_agentic_pipeline(user_message: str) -> dict:
    """
    Full multi-agent pipeline. Called once per user message.

    Step 1 : OrchestratorAgent decides the route.
    Step 2a : EditPlannerAgent builds tool steps  (edit_planner)
    Step 2b : ClarifierAgent builds clarify obj   (clarifier)

    """
    # Step 1: Orchestrate 
    orch = orchestrator_agent(user_message)
    action = orch.get("action")

    if action in ("undo", "reset"):
        return orch

    if action == "no_image":
        return {"action": "info", "message": orch.get("message", "Please upload an image first.")}

    if action == "error":
        return orch

    target = orch.get("target", "edit_planner")

    # Step 2a: Edit Planner 
    if target == "edit_planner":
        plan = edit_planner_agent(user_message)
        if "steps" in plan:
            return {
                "action": "execute",
                "steps": plan["steps"],
                "message": plan.get("message", ""),
            }
        return {"action": "error", "message": plan.get("message", "Edit planner failed.")}

    # Step 2b: Clarifier 
    if target == "clarifier":
        clarify = clarifier_agent(user_message)
        if "options" in clarify:
            return {"action": "clarify", **clarify}
        return {"action": "error", "message": "Clarifier failed to generate options."}

    return {"action": "error", "message": f"Unknown route target: {target}"}
