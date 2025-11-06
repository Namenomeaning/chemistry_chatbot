"""Test chemistry chatbot with image input."""

import time
from pathlib import Path
from src.agent.graph import graph
from src.agent.state import AgentState


def test_image_query(image_path: str, text_query: str = None, thread_id: str = "test-image"):
    """Test query with image input.

    Args:
        image_path: Path to chemical structure image
        text_query: Optional text query to accompany image
        thread_id: Thread ID for conversation context
    """
    print(f"\n{'='*80}")
    print(f"Image: {image_path}")
    if text_query:
        print(f"Text: {text_query}")
    print(f"{'='*80}")

    # Load image
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ Image not found: {image_path}")
        return None

    with open(image_file, "rb") as f:
        image_bytes = f.read()

    # Prepare input state
    initial_state = AgentState(
        input_text=text_query,
        input_image=image_bytes
    )

    # Run graph
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

    # Display results
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
            text_response = result['final_response'].get('text_response', '')
            print(f"  Text: {text_response[:200]}...")
            print(f"  Image: {result['final_response'].get('image_path', 'N/A')}")
            print(f"  Audio: {result['final_response'].get('audio_path', 'N/A')}")

    return result


def main():
    """Run comprehensive tests."""
    print("Chemistry Chatbot - Comprehensive Test Suite")
    print("⚠️  Wait 10 seconds between tests to avoid rate limits.\n")

    # Test 1: Image only (should return image + audio for overview)
    print("\n### Test 1: Image only - Ethanol structure ###")
    test_image_query(
        "src/data/images/ethanol.png",
        text_query=None,
        thread_id="test-1"
    )
    time.sleep(10)

    # Test 2: Text only - compound name (should return image + audio)
    print("\n\n### Test 2: Text only - Ask about methane ###")
    initial_state = AgentState(input_text="Cho tôi biết về methane", input_image=None)
    config = {"configurable": {"thread_id": "test-2"}}
    try:
        result = graph.invoke(initial_state, config)
        print(f"Rephrased: {result.get('rephrased_query', '')}")
        print(f"Search: {result.get('search_query', '')}")
        if result.get('final_response'):
            print(f"Response: {result['final_response'].get('text_response', '')[:200]}...")
            print(f"Image: {result['final_response'].get('image_path', 'N/A')}")
            print(f"Audio: {result['final_response'].get('audio_path', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    time.sleep(10)

    # Test 3: Ask only for formula (should return image only)
    print("\n\n### Test 3: Text only - Ask only for formula ###")
    initial_state = AgentState(input_text="Công thức của ethanol?", input_image=None)
    config = {"configurable": {"thread_id": "test-3"}}
    try:
        result = graph.invoke(initial_state, config)
        print(f"Rephrased: {result.get('rephrased_query', '')}")
        if result.get('final_response'):
            print(f"Response: {result['final_response'].get('text_response', '')[:200]}...")
            print(f"Image: {result['final_response'].get('image_path', 'N/A')}")
            print(f"Audio: {result['final_response'].get('audio_path', 'N/A')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    time.sleep(10)

    # Test 4: Image + text query
    print("\n\n### Test 4: Image + text - Acetic acid ###")
    test_image_query(
        "src/data/images/acetic_acid.png",
        text_query="Đây là hợp chất gì?",
        thread_id="test-4"
    )
    time.sleep(10)

    # Test 5: Follow-up with pronoun (conversation context)
    print("\n\n### Test 5: Follow-up - Methane then ask formula ###")
    test_image_query(
        "src/data/images/methane.png",
        text_query=None,
        thread_id="test-5"
    )
    time.sleep(10)

    print("\n### Test 5b: Follow-up with pronoun ###")
    initial_state = AgentState(input_text="Công thức của nó là gì?", input_image=None)
    config = {"configurable": {"thread_id": "test-5"}}
    try:
        result = graph.invoke(initial_state, config)
        print(f"Rephrased: {result.get('rephrased_query', '')}")
        print(f"Search: {result.get('search_query', '')}")
        if result.get('final_response'):
            print(f"Response: {result['final_response'].get('text_response', '')[:200]}...")
    except Exception as e:
        print(f"❌ Error: {e}")
    time.sleep(10)

    # Test 6: Invalid/unknown compound (should trigger error)
    print("\n\n### Test 6: Unknown compound (not in DB) ###")
    initial_state = AgentState(input_text="Cho tôi biết về benzene", input_image=None)
    config = {"configurable": {"thread_id": "test-6"}}
    try:
        result = graph.invoke(initial_state, config)
        print(f"RAG context: {len(result.get('rag_context', []))} docs")
        if result.get('final_response'):
            print(f"Response: {result['final_response'].get('text_response', '')[:200]}...")
            print(f"Selected doc_id: {result['final_response'].get('selected_doc_id', 'None')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    time.sleep(10)

    # Test 7: Non-chemistry question (should be rejected)
    print("\n\n### Test 7: Non-chemistry question ###")
    initial_state = AgentState(input_text="Thủ đô của Việt Nam là gì?", input_image=None)
    config = {"configurable": {"thread_id": "test-7"}}
    try:
        result = graph.invoke(initial_state, config)
        print(f"Chemistry-related: {result.get('is_chemistry_related', False)}")
        print(f"Error message: {result.get('error_message', 'None')}")
    except Exception as e:
        print(f"❌ Error: {e}")
    time.sleep(10)

    # Test 8: Invalid formula (should suggest correction)
    print("\n\n### Test 8: Invalid formula ###")
    initial_state = AgentState(input_text="Cho tôi biết về C2H5OX", input_image=None)
    config = {"configurable": {"thread_id": "test-8"}}
    try:
        result = graph.invoke(initial_state, config)
        print(f"Valid: {result.get('is_valid', False)}")
        print(f"Error message: {result.get('error_message', 'None')}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()