from .nodes import QUESTIONNAIRE
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, HumanMessage


def route_start(state):

    mode = state.get("start_route")
    plan = state.get("plan", None)
    
    if mode == "new_plan":
        return "new_plan"

    if not plan :
        return "load"

    return "discuss"

def route_after_question(state):
    if state["question_idx"] >= len(QUESTIONNAIRE):
        return "coach"
    return "continue"

def route_garmin(state):
    gar = state.get("garmin_consent")
    tav = state.get("search")

    if not gar and not tav :
        return "coach"
    
    return "search" if not gar else "garmin"

def route_tavily(state):
    tav = state.get("search")
    return "coach" if not tav else "search"
    
def route_modify(state):
    mode = state.get("modify_mode", "continue")
    return "continue" if mode == "continue" else "modify"




