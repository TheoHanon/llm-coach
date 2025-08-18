

import chainlit as cl
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import RunnableConfig
from my_coach.llm import init_llms
from my_coach.graph import build_graph


llms = init_llms()
graph = build_graph(llms)


@cl.on_chat_start
async def start():
    config = RunnableConfig({"configurable": {"thread_id": cl.context.session.id}})
    out = cl.Message("")
    await out.send()

    stream = graph.stream({"messages": [{"role" : "user", "content" : "Hello"}]}, stream_mode="messages", config=config)
    for msg_chunk, _ in stream:
        if isinstance(msg_chunk, AIMessage) and msg_chunk.content and msg_chunk.additional_kwargs.get("visible", True):
            await out.stream_token(str(msg_chunk.content))
    await out.update()


@cl.on_message
async def chat_stream(user_msg: cl.Message):
    config = RunnableConfig({"configurable": {"thread_id": cl.context.session.id}})
    out_msg = cl.Message("")
    await out_msg.send()

    stream = graph.stream(
        {"messages": [{"role": "user", "content": user_msg.content}]},
        stream_mode="messages",
        config=config,
    )

    for msg_chunk, _meta in stream:
        if isinstance(msg_chunk, AIMessage) and msg_chunk.content and msg_chunk.additional_kwargs.get("visible", True) and _meta["langgraph_node"] != "search":
            await out_msg.stream_token(str(msg_chunk.content))

    await out_msg.update()

