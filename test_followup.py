"""Test follow-up question handling."""

import time
from src.agent.graph import graph
from src.agent.state import AgentState


def test_followup():
    """Test follow-up question with conversation context."""
    thread_id = "followup-test"

    # First query
    print("="*80)
    print("First query: Cho tôi biết về methanol")
    print("="*80)

    state1 = AgentState(input_text="Cho tôi biết về methanol", input_image=None)
    config = {"configurable": {"thread_id": thread_id}}
    result1 = graph.invoke(state1, config)

    print(f"Rephrased query: {result1.get('rephrased_query', '')}")
    print(f"Chemistry-related: {result1.get('is_chemistry_related')}")
    print(f"Search query: {result1.get('search_query', '')}")
    print(f"Valid: {result1.get('is_valid')}")

    time.sleep(5)  # Flash Lite: 15 req/min = 4 seconds between requests

    # Follow-up query
    print("\n" + "="*80)
    print("Follow-up query: Còn công thức của nó?")
    print("="*80)

    state2 = AgentState(input_text="Còn công thức của nó?", input_image=None)
    result2 = graph.invoke(state2, config)

    print(f"Rephrased query: {result2.get('rephrased_query', '')}")
    print(f"Chemistry-related: {result2.get('is_chemistry_related')}")
    if result2.get('error_message'):
        print(f"Error: {result2['error_message']}")
    else:
        print(f"Search query: {result2.get('search_query', '')}")
        print(f"Valid: {result2.get('is_valid')}")


if __name__ == "__main__":
    test_followup()
