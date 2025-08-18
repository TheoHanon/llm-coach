from typing import Any, Dict
from langchain.chat_models import init_chat_model
from my_coach.config import settings
from my_coach.domain.shemas import WelcomeRoute, TrainingPlan, ModifyRoute


def init_llms() -> Dict[str, Any]:

    llm_small = init_chat_model(model = settings.model_small, model_provider=settings.model_provider)
    llm_large = init_chat_model(model = settings.model_large, model_provider=settings.model_provider)
    
    llm_coach = llm_large.with_structured_output(TrainingPlan)
    llm_route = llm_large.with_structured_output(WelcomeRoute)
    llm_modify = llm_large.with_structured_output(ModifyRoute)

    return {
        "llm_large" : llm_large, 
        "llm_small" : llm_small, 
        "llm_coach" : llm_coach,
        "llm_route" : llm_route, 
        "llm_modify" :llm_modify
    }