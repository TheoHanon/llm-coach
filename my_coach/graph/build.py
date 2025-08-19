from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition, ToolNode
from my_coach.tools_langchain.tool_load_training_plan import load_training_plan
from .state import State
from .routes import route_after_question, route_garmin, route_modify, route_start, route_tavily
from . import nodes
from typing import Any


# def _build_new_plan_graph(llms):

#     llm_small = llms["llm_small"]
#     llm_coach = llms["llm_coach"]

    
#     def garmin(state : State):
#         return nodes.garmin_node(state, llm_small)
    
#     def search(state:State):
#         return nodes.research_node(state, llm_small)
    
#     def coach(state: State):
#         return nodes.coach_node(state, llm_coach)
    
#     def summary(state: State):
#         return nodes.summary_node(state, llm_small)
    
#     def save_node(state: State):
#         return nodes.save_node(state, llm_small)

#     def save_confirm(state: State):
#         return nodes.save_confirm_node(state)
    
    
#     g = StateGraph(State)
#     g.add_node("garmin", garmin)
#     g.add_node("search", search)
#     g.add_node("coach", coach)
#     g.add_node("summary", summary)
#     g.add_node("save_node", save_node)
#     g.add_node("save_confirm", save_confirm)

#     g.add_conditional_edges(START, route_garmin, {"garmin" : "garmin", "search" : "search", "coach" : "coach"})
#     g.add_conditional_edges("garmin", route_tavily, {"coach" : "coach", "search" : "search"})
#     g.add_edge("garmin", "search")
#     g.add_edge("search", "coach")
#     g.add_edge("coach", "save_node")
#     g.add_edge("save_node", "save_confirm")
#     g.add_edge("save_confirm", "summary")

#     return g.compile()

# def build_graph(llms):

#     llm_small = llms["llm_small"]
#     llm_modify = llms["llm_modify"]


#     def questionnaire(state: State):
#         return nodes.questionnaire_node(state, llm_small)

#     def discuss(state : State):
#         return nodes.discuss_node(state, llm_small)
    
#     def modify(state : State):
#         return nodes.modify_node(state, llm_modify)
    
#     def load(state : State):
#         return nodes.load_node(state, llm_small)
    
#     new_plan = _build_new_plan_graph(llms)

#     g = StateGraph(State)
#     g.add_node("questionnaire", questionnaire)
#     g.add_node("discuss", discuss)
#     g.add_node("modify", modify)
#     g.add_node("new_plan", new_plan)
#     g.add_node("load", load)

#     g.add_conditional_edges(START, route_start, {
#         "discuss" : "modify", 
#         "new_plan" : "questionnaire",
#         "load" : "load"
#     })
    
#     g.add_conditional_edges("questionnaire", route_after_question, {
#         "continue" : END,
#         "coach" : "new_plan"
#     })
#     g.add_edge("new_plan", END)
#     g.add_edge("load", "discuss")
#     g.add_edge("discuss", END)

#     g.add_conditional_edges("modify", route_modify, {
#         "continue" : "discuss", 
#         "modify" : "new_plan"
#     })

#     ckp = MemorySaver()
#     return g.compile(checkpointer=ckp)
    
def build_graph(llms):
    llm_small = llms["llm_small"]
    llm_coach = llms["llm_coach"]
    llm_modify = llms["llm_modify"]

    # --- Node wrappers ---
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

    # Small passthrough to replace subgraph START routing
    def new_plan_entry(state: State):
        return {}

    # --- Build one big graph ---
    g = StateGraph(State)

    # top-level flow
    g.add_node("questionnaire", questionnaire)
    g.add_node("discuss", discuss)
    g.add_node("modify", modify)
    g.add_node("load", load)

    # former subgraph nodes (flattened)
    g.add_node("new_plan_entry", new_plan_entry)
    g.add_node("garmin", garmin)
    g.add_node("search", search)
    g.add_node("coach", coach)
    g.add_node("save_node", save_node)
    g.add_node("save_confirm", save_confirm)
    g.add_node("summary", summary)

    # START routing
    g.add_conditional_edges(START, route_start, {
        "discuss": "modify",
        "new_plan": "questionnaire",
        "load": "load",
    })

    # After questionnaire: either finish or enter new-plan path
    g.add_conditional_edges("questionnaire", route_after_question, {
        "continue": END,
        "coach": "new_plan_entry",
    })

    # New-plan entry: decide first step (garmin/search/coach)
    g.add_conditional_edges("new_plan_entry", route_garmin, {
        "garmin": "garmin",
        "search": "search",
        "coach": "coach",
    })

    # Garmin can jump straight to coach or go to search (kept default edge to search)
    g.add_conditional_edges("garmin", route_tavily, {
        "coach": "coach",
        "search": "search",
    })
    g.add_edge("garmin", "search")

    # Evidence → plan → save → confirm → summary
    g.add_edge("search", "coach")
    g.add_edge("coach", "save_node")
    g.add_edge("save_node", "save_confirm")
    g.add_edge("save_confirm", "summary")
    g.add_edge("summary", END)

    # Load path → discuss → END
    g.add_edge("load", "discuss")
    g.add_edge("discuss", END)

    # Modify path can either continue talking or jump into new-plan flow
    g.add_conditional_edges("modify", route_modify, {
        "continue": "discuss",
        "modify": "new_plan_entry",
    })

    return g.compile(checkpointer=MemorySaver())