import asyncio

from langchain_core.messages import HumanMessage, AIMessage

from niche_config import niche_factory
from src.state import VideoState
from src.workflow import build_workflow
import time


async def main() -> None:
    """
    Entry point for the Aether workflow.

    Runs an interactive chat loop that processes user
    input through the full workflow pipeline:
    message_analyzer -> search_node -> scraper.
    """
    print("Initializing Aether Workflow...")
    graph = build_workflow()

    # Initialize state with all required fields
    state: VideoState = {
        "niche": niche_factory("psychology"),
        "messages": [],
        "web_sources": [],
        "youtube_sources": [],
        "sources_overview": "",
        "intent": "",
        "use_search": False,
    }

    print("\n--- Chat Started. Type 'quit' or 'exit' to stop. ---")
    print("Propose an idea or ask to find trending topics.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue

        start = time.perf_counter()
        print("\nProcessing: analyzing, searching, scraping...")
        state["messages"].append(
            HumanMessage(content=user_input),
        )

        # Reset per-turn transient fields so they
        # don't carry over from previous turns.
        state["web_sources"] = []
        state["youtube_sources"] = []
        state["sources_overview"] = ""
        state["intent"] = ""
        state["use_search"] = False

        # Invoke the graph
        state = await graph.ainvoke(state)  # type: ignore

        # Debug output for the pipeline results
        print(f"\n  Intent:  {state['intent']}")
        print(f"  Web Sources: {state['web_sources']}")
        print(f"  YouTube Sources: {state['youtube_sources']}")
        overview = state["sources_overview"]
        if overview:
            print(f"  Overview: {overview}...")
        print()

        end = time.perf_counter()
        print(f"  Processing Time: {end - start:.2f} seconds")

        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            print(f"\nAether: {last_msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
