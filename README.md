# My Coach

Small assistant to create and discuss endurance training plans using [LangGraph](https://github.com/langchain-ai/langgraph) and [Chainlit](https://docs.chainlit.io/).

## Install
Clone the repo and, from the **root directory**, run:
```bash
pip install -e .
````
## Run

In the root directory setup a `.env` file with your MISTRAL and TAVILY API KEY.
```bash
MISTRAL_API_KEY="YOUR KEY"
TAVILY_API_KEY="YOUR KEY"
```
Also from the **root directory**, start the app with:
```bash
chainlit run my_coach/ui/chainlit_app.py
```

## Notes

* Requires Python 3.10+
* `pip install -e .` is needed so relative imports work


