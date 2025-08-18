from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from my_coach.tools_langchain.tool_load_training_plan import load_training_plan
from .state import State
from .routes import route_after_question, question_or_welcome_or_discuss, route_modify
from . import nodes
from typing import Any



def build_graph(llms):
    llm_small = llms["llm_small"]
    llm_large = llms["llm_large"]
    llm_coach = llms["llm_coach"]
    llm_route = llms["llm_route"]
    llm_modify = llms["llm_modify"]

    def welcome(state: State):
        return nodes.welcome_node(state, llm_small)

    def questionnaire(state: State):
        return nodes.questionnaire_node(state, llm_small)

    def coach(state: State):
        return nodes.coach_node(state, llm_coach)

    def summary(state: State):
        return nodes.summary_node(state, llm_small)

    def save_node(state: State):
        return nodes.save_node(state, llm_small)

    def save_confirm(state: State):
        return nodes.save_confirm_node(state)
    
    def after_welcome(state : State):
        return nodes.after_welcome_node(state, llm_route)
    
    def discuss(state : State):
        return nodes.discuss_node(state, llm_small.bind_tools([load_training_plan]))
    
    def search(state:State):
        return nodes.research_node(state, llm_small)
    
    def modify(state : State):
        return nodes.modify_node(state, llm_modify)
    
    def garmin(state : State):
        return nodes.garmin_node(state, llm_small)
    

    g = StateGraph(State)
    g.add_node("discuss", discuss)
    g.add_node("load_node", nodes.load_node)
    g.add_node("welcome", welcome)
    g.add_node("after_welcome", after_welcome)
    g.add_node("questionnaire", questionnaire)
    g.add_node("garmin", garmin)
    g.add_node("coach", coach)
    g.add_node("summary", summary)
    g.add_node("save_node", save_node)
    g.add_node("save_confirm", save_confirm)
    g.add_node("search", search)
    g.add_node("modify", modify)


    g.add_conditional_edges(START, question_or_welcome_or_discuss, {"questionnaire": "questionnaire", "welcome": "welcome", "discuss" : "modify", "after_welcome" : "after_welcome"})
    g.add_edge("welcome", END)
    g.add_conditional_edges("questionnaire", route_after_question, {"coach": "search", "continue": END})
    g.add_conditional_edges("after_welcome", question_or_welcome_or_discuss, {"questionnaire": "questionnaire", "welcome": "welcome", "discuss" : "discuss", "after_welcome" : "after_welcome"})
    g.add_edge("search", "garmin")
    g.add_edge("garmin", "coach")
    g.add_edge("coach", "summary")
    g.add_edge("summary", "save_node")
    g.add_edge("save_node", "save_confirm")
    g.add_edge("save_confirm", END)
    g.add_conditional_edges("modify", route_modify, {"continue" : "discuss", "modify" : "search"})
    g.add_conditional_edges("discuss", tools_condition, {"tools" : "load_node", "__end__": END})
    g.add_edge("load_node", "discuss")


    ckp = MemorySaver()
    return g.compile(checkpointer=ckp)