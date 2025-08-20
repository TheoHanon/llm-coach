import os
import chainlit as cl
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.graph import START
from langchain_core.runnables.config import RunnableConfig
from my_coach.llm import init_llms
from my_coach.graph import build_graph


llms = init_llms()
graph = build_graph(llms)


def _make_config():
    return RunnableConfig({"configurable": {"thread_id": cl.context.session.id}})


async def _run_graph(entry: str):
    """Kick off the graph exactly once with a chosen entry."""
    config = _make_config()
    out = cl.Message("")
    await out.send()

    # We set start_route here; no need to send a fake "Hello"
    graph.update_state(config, values={"start_route": entry}, as_node=START)
    stream = graph.stream(
        {"messages": ["Hello"]}, stream_mode="messages", config=config
    )

    for msg_chunk, meta in stream:
        if (
            isinstance(msg_chunk, AIMessage)
            and msg_chunk.content
            and msg_chunk.additional_kwargs.get("visible", True)
        ):
            if not meta or meta.get("langgraph_node") != "search":
                await out.stream_token(str(msg_chunk.content))
    await out.update()


@cl.on_chat_start
async def start():
    config = _make_config()
    tav = True if os.environ.get("TAVILY_API_KEY") else False
    gar = True if os.environ.get("GARTH_TOKEN") else False

    graph.update_state(config=config, values={"search": tav, "garmin_consent": gar})

    res = await cl.AskActionMessage(
        content=(
            "Hello there, awesome athlete! 🏃‍♂️🚴‍♀️\nReady to begin? Choose an option:"
        ),
        actions=[
            cl.Action(
                label="Make a new plan with the Coach",
                payload={"value": "new_plan"},
                name="action_button",
            ),
            cl.Action(
                label="Modify your plan with the Coach",
                payload={"value": "discuss"},
                name="action_button",
            ),
        ],
    ).send()

    if res and res.get("payload") and res["payload"].get("value"):
        await _run_graph(res["payload"]["value"])


@cl.on_message
async def chat_stream(user_msg: cl.Message):
    config = _make_config()
    out_msg = cl.Message("")
    await out_msg.send()

    stream = graph.stream(
        {"messages": [{"role": "user", "content": user_msg.content}]},
        stream_mode="messages",
        config=config,
    )

    for msg_chunk, _meta in stream:
        if (
            isinstance(msg_chunk, AIMessageChunk)
            and msg_chunk.content
            and msg_chunk.additional_kwargs.get("visible", True)
            and _meta["langgraph_node"] != "search"
        ):
            await out_msg.stream_token(str(msg_chunk.content))

        if isinstance(msg_chunk, AIMessage) and not isinstance(
            msg_chunk, AIMessageChunk
        ):
            await out_msg.stream_token("\n\n")

    await out_msg.update()
