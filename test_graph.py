"""Test script for chemistry chatbot graph."""

import time
from src.agent.graph import graph
from src.agent.state import AgentState


def test_query(query: str, thread_id: str = "test-1"):
    """Test a single query through the graph.

    Args:
        query: User query text
        thread_id: Thread ID for conversation context
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")

    # Prepare input state
    initial_state = AgentState(
        input_text=query,
        input_image=None
    )

    # Run graph with checkpointer and measure time
    config = {"configurable": {"thread_id": thread_id}}
    start_time = time.time()

    try:
        result = graph.invoke(initial_state, config)
        elapsed_time = time.time() - start_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Error during graph execution: {e}")
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("\n⚠️  Rate limit exceeded! Please wait 60 seconds and try again.")
            print("💡 Tip: Consider upgrading to a paid Gemini API plan for higher limits.")
        return None

    # Display results (result is a dict from LangGraph)
    print(f"\n⏱️  Time: {elapsed_time:.2f}s")
    print(f"\nRephrased query: {result.get('rephrased_query', '')}")
    print(f"Chemistry-related: {result.get('is_chemistry_related', False)}")
    if result.get('error_message'):
        print(f"Error: {result['error_message']}")
    else:
        print(f"Search query: {result.get('search_query', '')}")
        print(f"Valid: {result.get('is_valid', False)}")
        if result.get('rag_context'):
            print(f"\nRAG context ({len(result['rag_context'])} docs):")
            for i, doc in enumerate(result['rag_context'], 1):
                print(f"  {i}. {doc.get('iupac_name', 'N/A')} - {doc.get('formula', 'N/A')}")
        if result.get('final_response'):
            print(f"\nFinal response:")
            print(f"  Text: {result['final_response'].get('text_response', '')[:100]}...")
            print(f"  Image: {result['final_response'].get('image_path', 'N/A')}")
            print(f"  Audio: {result['final_response'].get('audio_path', 'N/A')}")

    return result


def main():
    """Run test cases."""
    print("Chemistry Chatbot Graph - End-to-End Test")
    print("Note: Rate limit delay of 10 seconds between queries (Flash Lite: 15 req/min)\n")
    print("⚠️  Each query makes multiple API calls, so we need generous delays.\n")

    # Test 1: Non-chemistry query (should stop at check_relevance)
    print("\n\n### Test 1: Non-chemistry query ###")
    test_query("Hôm nay thời tiết thế nào?", thread_id="test-1")
    time.sleep(10)  # Flash Lite: 15 req/min, each query makes 2-3 API calls

    # Test 2: Invalid formula (should stop at extract_and_validate)
    print("\n\n### Test 2: Invalid formula ###")
    test_query("Cho tôi biết về C2H5OX", thread_id="test-2")
    time.sleep(10)

    # Test 3: Valid query - exact match
    print("\n\n### Test 3: Valid query - Ethanol ###")
    test_query("Cho tôi biết về ethanol", thread_id="test-3")
    time.sleep(10)

    # Test 4: Valid query - Vietnamese name
    print("\n\n### Test 4: Valid query - Vietnamese name ###")
    test_query("Rượu etylic là gì?", thread_id="test-4")
    time.sleep(10)

    # Test 5: Follow-up question (uses checkpointer context)
    print("\n\n### Test 5: Follow-up question ###")
    test_query("Cho tôi biết về methanol", thread_id="test-5")
    time.sleep(10)
    test_query("Còn công thức của nó?", thread_id="test-5")  # Should use context


if __name__ == "__main__":
    main()
