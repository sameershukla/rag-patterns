import os

from langchain_anthropic import ChatAnthropic
from langchain_core.chat_history import InMemoryChatMessageHistory


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Please set ANTHROPIC_API_KEY before running this script.")

    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0
    )

    # In-memory chat history for the current session
    chat_history = InMemoryChatMessageHistory()
    print("Basic memory chat started. Type 'exit' to quit.\n")
    while(True):
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("\n Chat ended")
            break

        if not user_input:
            print("Please enter a message:")
            continue

        #Store messages
        chat_history.add_ai_message(user_input)
        response = llm.invoke(chat_history.messages)
        print(f"\n AI {response.content}\n")

if __name__ == "__main__":
    main()
