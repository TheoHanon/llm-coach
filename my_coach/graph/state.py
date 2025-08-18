from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from langgraph.graph import add_messages
from functools import partial


def add_and_trim(prev : list, new : list, k : int):
    return add_messages(prev, new)

add_and_trim8 = partial(add_and_trim, k = 8)

class State(TypedDict):

    messages : Annotated[list, add_and_trim8]
    plan : Optional[List]
    justification : Optional[str]
    specs : Dict[str, Any]
    question_idx : Optional[int]
    welcome : Optional[bool]
    mode : Literal["make", "discuss"]
    modify_mode : Literal["modify", "continue"]
    modify_query : str
    web_ctx : Optional[str]
    garmin_data : Optional[str]
    garmin_consent : bool



    
