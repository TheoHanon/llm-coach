from typing import TypedDict, Annotated, Optional, List, Dict, Any, Literal
from langgraph.graph import add_messages
from langgraph.channels import LastValue
from functools import partial


def add_and_trim(prev : list, new : list, k : int):
    return add_messages(prev, new)

add_and_trim8 = partial(add_and_trim, k = 8)

class State(TypedDict):

    start_route : Annotated[Optional[Literal["discuss", "new_plan"]], LastValue]
    modify_mode : Literal["modify", "continue"]
    search : Optional[bool]
    garmin_consent : Optional[bool]


    messages : Annotated[list, add_and_trim8]
    plan : Optional[List]
    justification : Optional[str]
    specs : Dict[str, Any]
    question_idx : Optional[int]
    welcome : Optional[bool]
    mode : Literal["make", "discuss"]
    modify_query : str
    web_ctx : Optional[str]
    garmin_data : Optional[str]



    
