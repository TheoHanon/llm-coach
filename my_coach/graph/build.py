from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .state import State
from .routes import route_after_question, route_garmin, route_modify, route_start, route_tavily
from . import nodes
from typing import Any

    
def build_graph(llms):
    llm_small = llms["llm_small"]
    llm_coach = llms["llm_coach"]
    llm_modify = llms["llm_modify"]

    def questionnaire(state: State):
        return nodes.questionnaire_node(state, llm_small)

    def discuss(state: State):
        return nodes.discuss_node(state, llm_small)

    def modify(state: State):
        return nodes.modify_node(state, llm_modify)

    def load(state: State):
        return nodes.load_node(state, llm_small)

    def garmin(state: State):
        return nodes.garmin_node(state, llm_small)

    def search(state: State):
        return nodes.research_node(state, llm_small)

    def coach(state: State):
        return nodes.coach_node(state, llm_coach)

    def save_node(state: State):
        return nodes.save_node(state, llm_small)

    def save_confirm(state: State):
        return nodes.save_confirm_node(state)

    def summary(state: State):
        return nodes.summary_node(state, llm_small)

    def new_plan_entry(state: State):
        return {}
    
    def retriever(state : State):
        return nodes.retriever_node(state, llm_small)

    
    g = StateGraph(State)

    
    g.add_node("questionnaire", questionnaire)
    g.add_node("discuss", discuss)
    g.add_node("modify", modify)
    g.add_node("load", load)


    g.add_node("new_plan_entry", new_plan_entry)
    g.add_node("garmin", garmin)
    g.add_node("retriever", retriever)
    g.add_node("search", search)
    g.add_node("coach", coach)
    g.add_node("save_node", save_node)
    g.add_node("save_confirm", save_confirm)
    g.add_node("summary", summary)

    
    g.add_conditional_edges(START, route_start, {
        "discuss": "modify",
        "new_plan": "questionnaire",
        "load": "load",
    })


    g.add_conditional_edges("questionnaire", route_after_question, {
        "continue": END,
        "coach": "new_plan_entry",
    })

    
    g.add_conditional_edges("new_plan_entry", route_garmin, {
        "garmin": "garmin",
        "search": "search",
        "coach": "retriever",
    })


    g.add_conditional_edges("garmin", route_tavily, {
        "coach": "retriever",
        "search": "search",
    })
    g.add_edge("garmin", "search")

    
    g.add_edge("search", "retriever")
    g.add_edge("retriever", "coach")
    g.add_edge("coach", "save_node")
    g.add_edge("save_node", "save_confirm")
    g.add_edge("save_confirm", "summary")
    g.add_edge("summary", END)

    
    g.add_edge("load", "discuss")
    g.add_edge("discuss", END)

    
    g.add_conditional_edges("modify", route_modify, {
        "continue": "discuss",
        "modify": "new_plan_entry",
    })

    return g.compile(checkpointer=MemorySaver())