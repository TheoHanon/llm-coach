from typing import List, Dict
from datetime import datetime, timedelta
import json
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from my_coach.tools_langchain.tool_save_training_plan import save_training_plan
from my_coach.tools_langchain.tool_load_training_plan import load_training_plan
from my_coach.tools_langchain.tool_search import tool_search
from my_coach.mcp.mcp_garmin import garmin_tool
from .utils import _get_fitness_summary, _build_query, _retrieve

QUESTIONNAIRE: Dict[str, str] = {
    "sport": "The sport (running / cycling / trail / triathlon) you want a program for",
    "goal": "Goal (e.g., finish, 10k in 45:00, build base, comeback). If no event, plan length (weeks, e.g., 8 or 12):",
    "target_event_date": "Target event date:",
    "current_weekly_volume": "Current weekly volume:",
    "longest_recent": "Longest recent session:",
    "weekly_availability": "Weekly availability (days + approx duration):",
    "constraints": "Constraints (injuries, travel, equipment, surfaces)",
    "additional_remarks": "Additional specification the user might want to work on ?",
}
FIELDS: List[str] = [
    "sport",
    "goal",
    "target_event_date",
    "current_weekly_volume",
    "longest_recent",
    "weekly_availability",
    "constraints",
    "additional_remarks",
]


def retriever_node(state, llm):
    specs = state.get("specs", {}) or {}
    modify_query = state.get("modify_query") or None
    garmin_data = (state.get("garmin") or "").strip()

    # simple formatter
    parts = []
    for k, v in specs.items():
        if v:
            parts.append(f"{k}: {v}")

    if garmin_data:
        parts.append(f"garmin data: {garmin_data}")

    if modify_query:
        parts.append("additional request: " + " ".join(modify_query))

    q = " | ".join(parts)
    _, bib, ctx = _retrieve(q, k=4)

    rag_ctx = {"brief": ctx, "sources": bib}

    return {"rag_ctx": rag_ctx}


def research_node(state, llm):
    specs = state.get("specs", {})
    modify_query = state.get("modify_query", "")

    if specs.get("additional_remarks"):
        specs["additional_remarks"] += f"\n {modify_query}"

    query = _build_query(specs)

    # search

    try:
        hits = tool_search.invoke({"query": query})
    except Exception:
        hits = None

    results = (hits or {}).get("results", [])

    sys = (
        "You compress web results into a small, strictly grounded brief for a coach.\n"
        "Rules:\n"
        "- Only use the provided snippets.\n"
        "- Output <= 10 bullets.\n"
        "- Each bullet: actionable facts or protocols worth knowing while creating a training plan, followed by (source: URL).\n"
        "- Prefer consensus/guidelines; avoid fringe claims."
    )
    pack = (
        "\n\n".join(
            f"TITLE: {r.get('title')}\nURL: {r.get('url')}\nEXCERPT:\n{r.get('content', '')[:1500]}"
            for r in results
        )
        or "No results."
    )

    brief = llm.invoke([SystemMessage(content=sys), HumanMessage(content=pack)])

    web_ctx = {
        "brief": brief.content,
        "sources": [{"title": r.get("title"), "url": r.get("url")} for r in results],
    }

    return {"web_ctx": web_ctx}


def discuss_node(state, llm):
    plan = state.get("plan", [])
    sys = (
        "You are a professional endurance coach (running, cycling, trail, triathlon). "
        "Ask what the user want to modify to its current training plan."
        "Your role:\n"
        "- If the plan is available, give concise, constructive feedback on the plan.\n"
        "- Be supportive and professional, not too verbose (max 2–3 sentences unless details are needed).\n"
        "- Keep tone positive and coaching-like. Use at most one emoji occasionally.\n"
        "======\n"
        f"THE PLAN:\n {plan}"
    )

    resp = llm.invoke([SystemMessage(content=sys), *state["messages"]])
    resp.additional_kwargs["visible"] = False

    return {"messages": [resp]}


def questionnaire_node(state, llm):
    sys = (
        "Your role is simply to rewrite the question in a nice way for the user. \n"
        "Stay concise but not cold. The questions will tailor a training plan. Ensure continuity.\n"
        "You must only ask one question at the time.\n"
        "Strictly stick to the provided question.\n"
        "Even if the user accidentaly answered the question already ask it again.\n"
        "Not need to say hi to the user."
        ""
    )

    step = state.get("question_idx", 0)
    specs = state.get("specs", {})
    garmin_consent = state.get("garmin_consent")

    if garmin_consent and step == 0:
        QUESTIONNAIRE.pop("current_weekly_volume", None)
        QUESTIONNAIRE.pop("longest_recent", None)
        FIELDS.remove("current_weekly_volume")
        FIELDS.remove("longest_recent")

    if step < len(QUESTIONNAIRE) and step != 0:
        usr_resp = state["messages"][-1].content
        specs[FIELDS[step - 1]] = usr_resp
        step += 1

    if step < len(FIELDS):
        nf = FIELDS[step]
        generic_q = QUESTIONNAIRE[nf]

        resp = llm.invoke(
            input=[
                SystemMessage(content=sys),
                *state["messages"],
                HumanMessage(content="QUESTION:\n" + generic_q),
            ]
        )

        new_q = resp.content

        return {
            "messages": [
                AIMessage(content=new_q, additional_kwargs={"visible": False})
            ],
            "question_idx": step + 1 if step == 0 else step,
            "specs": specs,
        }

    else:
        return {
            "question_idx": step,
            "specs": specs,
        }


def garmin_node(state, llm):
    try:
        end = datetime.today().date()
        start = end - timedelta(days=90)
        payload = garmin_tool.invoke({"from_date": start, "to_date": end})
        resp = json.loads(payload)

        summary = _get_fitness_summary(resp)

    except Exception as e:
        raise ValueError(f"Error while loading Garmin data : {e}")

    sys = (
        "You are an endurance coach. Given a short JSON of the last 90 days of Garmin data, "
        "write 3–5 sentences that explain what it means for fitness and training readiness. "
        "Be clear, motivational, and avoid restating numbers verbatim. If empty, say so."
    )

    hum = f"Here is the summary JSON:\n{summary}"

    brief = llm.invoke([SystemMessage(content=sys), HumanMessage(content=hum)])
    brief.additional_kwargs["visible"] = False

    return {"garmin_data": summary, "messages": [brief]}


def coach_node(state, llm):
    specs_blob = json.dumps(state.get("specs") or {}, ensure_ascii=False)
    web_brief = state.get("web_ctx", {}).get("brief", "")
    rag_ctx = state.get("rag_ctx", {}).get("brief", "")
    modify_query = state.get("modify_query")
    garmin = state.get("garmin_data")

    MAX_RAG_CHARS = 8000
    MAX_WEB_BRIEF = 2000
    rag_ctx = rag_ctx[:MAX_RAG_CHARS]
    web_brief = web_brief[:MAX_WEB_BRIEF]

    sys = (
        f"Today is {datetime.today():%Y-%m-%d}.\n"
        "You are a professional endurance coach.\n"
        "Populate the structured fields the caller requested; do not add extra keys or any commentary outside the structured output.\n"
        "STRICT : Respect availability of the user , current volume, constraints; progress conservatively.\n"
        "SCRICT GUIDELINE : Generate at most 28 sessions. \n"
        "Use evidence from RAG and web when helpful; place any citations only in the justification (e.g., [1], [W1]).\n"
        "DESCRIPTION: one medium line (≈120-200 chars, 1-2 sentences, no newlines). Include: sport; warm-up; main set; cool-down; intensity (pace/power/HR/RPE/temp). Avoid bare lines like “60 min Z1”.\n"
        "Rest day don't need to be stated."
    )

    hum = (
        "--- TRAINING SPEC (JSON) ---\n"
        + specs_blob
        + (f"\n--- GARMIN (bounds) ---\n{garmin}" if garmin else "")
        + "\n--- EVIDENCE CONTEXT (RAG) ---\n"
        + (rag_ctx[:8000] or "No local evidence.")
        + "\n--- WEB BRIEF ---\n"
        + (web_brief or "No web brief.")
    )

    if modify_query:
        hum += "\n--- USER MODIFY REQUEST ---\n" + "|".join(modify_query) + "\n"

    resp = llm.invoke([SystemMessage(content=sys), HumanMessage(content=hum)])

    return {
        "plan": getattr(resp, "plan", state.get("plan", [])),
        "justification": getattr(resp, "justification", "No justification provided."),
        "modify_mode": "continue",
    }


def summary_node(state, llm):
    sys = (
        "Print the training plan as a clean markdown table, then add a 1–2 line justification. "
        "End with a compact 'Sources' list using markdown links."
        "Cite all the provided sources."
    )

    web_sources = (state.get("web_ctx")).get("sources", [])
    rag_sources = state.get("rag_ctx", {}).get("sources", [])

    hum = json.dumps(
        {
            "plan": [item.model_dump_json() for item in state["plan"]],
            "justification": state.get("justification"),
            "sources": [web_sources, rag_sources],
        },
        ensure_ascii=False,
    )

    resp = llm.invoke(input=[SystemMessage(content=sys), HumanMessage(content=hum)])
    resp.additional_kwargs["visible"] = False

    return {"messages": [resp], "start_route": "discuss"}


def load_node(state, llm):
    try:
        result = load_training_plan.invoke({})
        data = json.loads(result)

        text = (
            "✅ Plan loaded"
            if data.get("status") == "ok"
            else f"⚠️ Save error: {data.get('error')}"
        )

    except Exception as e:
        data = {}
        text = f"⚠️ Plan saving failed: {e}"

    return {
        "messages": [
            AIMessage(content="\n\n" + text, additional_kwargs={"visible": False}),
            HumanMessage(
                content="Shows me (markdown table plz) and comments my plan plz"
            ),
        ],
        "plan": data.get("plan", []),
    }


def save_node(state, llm):
    try:
        result = save_training_plan.invoke({"training_plan": state["plan"]})
        data = json.loads(result)
        text = (
            (
                f"✅ Plan saved ({data['rows_written']} rows) to `{data['path']}` "
                f"[{data['date_range']['start']} → {data['date_range']['end']}]"
            )
            if data.get("status") == "ok"
            else f"⚠️ Save error: {data}"
        )
    except Exception as e:
        text = f"⚠️ Plan saving failed: {e}"

    return {
        "messages": [
            AIMessage(content="\n\n" + text, additional_kwargs={"visible": True})
        ]
    }


def modify_node(state, llm):
    sys = (
        "You are a strict router. Output ONLY the structured object.\n"
        "- 'modify' ONLY if the user EXPLICITLY asks to change the existing plan/schedule/sessions.\n"
        "- If the request is a general question, praise, or unrelated, return 'continue'."
    )

    last_usr_msg = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            last_usr_msg = m.content
            break

    resp = llm.invoke(
        [SystemMessage(content=sys), HumanMessage(content=last_usr_msg)],
        config={"temperature": 0, "max_tokens": 150},
    )

    return {"modify_mode": resp.mode, "modify_query": [last_usr_msg]}


def save_confirm_node(state):
    msg = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None
    )
    text = "Saved your plan."
    if msg and msg.content:
        try:
            data = json.loads(str(msg.content))
            if data.get("status") == "ok":
                text = (
                    f"✅ Plan saved ({data['rows_written']} rows) to `{data['path']}` "
                    f"[{data['date_range']['start']} → {data['date_range']['end']}]"
                )
            else:
                text = f"⚠️ Save error: {data}"
        except Exception:
            text = ""

    return {
        "messages": [
            AIMessage(content="\n\n" + text, additional_kwargs={"visible": True})
        ]
    }
