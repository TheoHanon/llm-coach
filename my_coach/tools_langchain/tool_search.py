from langchain_tavily import TavilySearch


tool_search = TavilySearch(
    max_result = 3,
    topic = "general"
)