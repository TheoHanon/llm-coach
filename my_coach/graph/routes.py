from .nodes import QUESTIONNAIRE
from langchain_core.messages import ToolMessage, AIMessage, SystemMessage, HumanMessage


def route_after_question(state):
    if state["question_idx"] >= len(QUESTIONNAIRE):
        return "coach"
    return "continue"


def question_or_welcome_or_discuss(state):
    mode = state.get("mode", None)
    welcome = state.get("welcome", None)
    
    if welcome:
        if not mode:
            return "after_welcome"
        elif mode == "discuss":
            return "discuss"
        else:
            return "questionnaire"
    else:
        return "welcome"
    
def route_modify(state):
    mode = state.get("modify_mode", "continue")
    return "continue" if mode == "continue" else "modify"




