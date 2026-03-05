from langgraph.graph import StateGraph
from src.state import VideoState
from src.nodes.trending_research import trending_research

def build_workflow():
    graph = StateGraph(VideoState)
    graph.add_node("trending_research", trending_research)
    graph.set_entry_point("trending_research")
    graph.set_finish_point("trending_research")
    return graph.compile()
