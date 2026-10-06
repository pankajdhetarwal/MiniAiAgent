import json
from config import openai_client, OPENAI_MODEL
from tools import vector_search_tool, rerank_results, calculator_tool
from memory import (
    store_chat_message,
    retrieve_session_history,
)


# Define a tool selector function that decides which tool to use based on user input and message history
def tool_selector(user_input, session_history=None):
    messages = [
        {
            "role": "system",
            "content": (
                "Select the appropriate tool from the options below. Consider the full context of the conversation before deciding.\n\n"
                "Tools available:\n"
                "- vector_search_tool: Retrieve specific context about recent MongoDB earnings and announcements\n"
                "- calculator_tool: For mathematical operations\n"
                "- none: For general questions without additional context\n"
                "Process for making your decision:\n"
                "1. Analyze if the current question relates to or follows up on a previous vector search query\n"
                "2. For follow-up questions, incorporate context from previous exchanges to create a comprehensive search query\n"
                "3. Only use calculator_tool for explicit mathematical operations\n"
                "4. Default to none only when certain the other tools won't help\n\n"
                "When continuing a conversation:\n"
                "- Identify the specific topic being discussed\n"
                "- Include relevant details from previous exchanges\n"
                "- Formulate a query that stands alone but preserves conversation context\n\n"
                'Return a JSON object only: {"tool": "selected_tool", "input": "your_query"}\n'
                "You must always return both keys, 'tool' and 'input', with non-null string values. Never omit a key or return null.\n\n"
                "Examples:\n"
                'Question: "What was MongoDB\'s total revenue last quarter?"\n'
                'Return: {"tool": "vector_search_tool", "input": "MongoDB total revenue last quarter"}\n\n'
                'Question: "What is 45 times 12?"\n'
                'Return: {"tool": "calculator_tool", "input": "45 * 12"}\n\n'
                'Question: "Hi, how are you?"\n'
                'Return: {"tool": "none", "input": "Hi, how are you?"}\n'
            ),
        }
    ]
    if session_history:
        messages.extend(session_history)
    messages.append({"role": "user", "content": user_input})

    response = (
        openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type": "json_object"},  # Ollama's grammar-constrained decoder enforces real JSON
        )
        .choices[0]
        .message.content
    )
    try:
        tool_call = json.loads(response)
        return tool_call.get("tool"), tool_call.get("input")
    except Exception as e:
        print(f"[tool_selector] parse failed: {e!r}")
        print(f"[tool_selector] raw response: {response!r}")
        return "none", user_input


# Define the agent workflow
def generate_response(session_id: str, user_input: str) -> str:

    # Store the user input in the chat history collection
    store_chat_message(session_id, "user", user_input)

    # Initialize a list of inputs to pass to the LLM
    llm_input = []

    # Retrieve the session history for the current session and add it to the LLM input
    session_history = retrieve_session_history(session_id)
    llm_input.extend(session_history)

    # Append the user message in the correct format
    user_message = {"role": "user", "content": user_input}
    llm_input.append(user_message)

    # Call the tool_selector function to determine which tool to use
    tool, tool_input = tool_selector(user_input, session_history)
    print("Tool selected: ", tool)

    # Process based on selected tool
    if tool == "vector_search_tool":
        # Step 1: Vector search retrieves top 20 candidates
        raw_results = vector_search_tool(tool_input)
        print(f"  Vector search returned {len(raw_results)} candidates")

        # Step 2: Cohere reranker narrows to top 5
        context = rerank_results(tool_input, raw_results, top_n=5)
        print(f"  Reranked to top {len(context)} results")

        system_message_content = (
            f"Answer the user's question based on the retrieved context and conversation history.\n"
            f"1. First, understand what specific information the user is requesting\n"
            f"2. Then, locate the most relevant details in the context provided\n"
            f"3. Finally, provide a clear, accurate response that directly addresses the question\n\n"
            f"If the current question builds on previous exchanges, maintain continuity in your answer.\n"
            f"Only state facts clearly supported by the provided context. If information is not available, say 'I DON'T KNOW'.\n\n"
            f"Context:\n{context}"
        )
        response = get_llm_response(llm_input, system_message_content)
    elif tool == "calculator_tool":
        response = calculator_tool(tool_input)
    else:
        system_message_content = "You are a helpful assistant. Respond to the user's prompt as best as you can based on the conversation history."
        response = get_llm_response(llm_input, system_message_content)

    # Store the system response in the chat history collection
    store_chat_message(session_id, "assistant", response)
    return response


# Helper function to get the LLM response
def get_llm_response(messages, system_message_content):
    system_message = {
        "role": "system",
        "content": system_message_content,
    }

    # Work on a copy so we don't mutate the caller's list
    messages = list(messages)

    if any(msg.get("role") == "system" for msg in messages):
        messages.append(system_message)
    else:
        messages = [system_message] + messages

    response = (
        openai_client.chat.completions.create(model=OPENAI_MODEL, messages=messages)
        .choices[0]
        .message.content
    )

    return response