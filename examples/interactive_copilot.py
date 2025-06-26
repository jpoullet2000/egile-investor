"""
Example usage of the CopilotMCPAgent - GitHub Copilot style interaction
"""

import asyncio
from egile_investor.copilot_mcp_agent import CopilotMCPAgent


async def demo_copilot_interaction():
    """Demonstrate conversational interaction with the Copilot agent."""

    # Create a copilot session using async context manager
    async with CopilotMCPAgent(agent_name="Investment Copilot") as copilot:
        print(
            "🤖 Investment Copilot: Hi! I'm here to help with investment analysis and market research."
        )
        print(
            "     You can ask me about stocks, create portfolios, screen for opportunities, and more!\n"
        )

        # Simulate a conversation
        conversations = [
            "What can you help me with?",
            "I'm interested in tech stocks. Can you find some good options?",
            "Tell me more about the first stock you found",
            "What are the risks with that stock?",
            "Can you compare it with NVIDIA?",
            "Should I invest in it?",
        ]

        for user_input in conversations:
            print(f"👤 User: {user_input}")
            response = await copilot.chat(user_input)
            print(f"🤖 Investment Copilot: {response}\n")

            # Small delay to simulate natural conversation
            await asyncio.sleep(1)

        # Show conversation history
        print("📜 Conversation Summary:")
        history = copilot.get_conversation_history()
        for i, msg in enumerate(history[-6:], 1):  # Last 3 exchanges
            role = "User" if msg["role"] == "user" else "Copilot"
            content = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )
            print(f"{i}. {role}: {content}")


async def interactive_copilot_session():
    """Interactive session where you can chat with the copilot."""

    async with CopilotMCPAgent(agent_name="Investment Copilot") as copilot:
        print(
            "🤖 Investment Copilot: Hello! I'm ready to help with investment analysis."
        )
        print(
            "     Type 'quit' to exit, 'history' to see conversation, 'clear' to start fresh, or 'suggest' for ideas.\n"
        )

        while True:
            try:
                user_input = input("👤 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("🤖 Investment Copilot: Goodbye! Happy investing! 👋")
                    break
                elif user_input.lower() == "history":
                    history = copilot.get_conversation_history()
                    print("\n📜 Conversation History:")
                    for msg in history[-10:]:  # Last 5 exchanges
                        role = "You" if msg["role"] == "user" else "Copilot"
                        print(f"   {role}: {msg['content'][:100]}...")
                    print()
                    continue
                elif user_input.lower() == "clear":
                    copilot.clear_conversation()
                    print(
                        "🤖 Investment Copilot: Conversation cleared! What would you like to explore?\n"
                    )
                    continue
                elif user_input.lower() == "suggest":
                    suggestions = await copilot.suggest_next_steps()
                    print(f"🤖 Investment Copilot: {suggestions}\n")
                    continue
                elif not user_input:
                    continue

                print("🤖 Investment Copilot: ", end="", flush=True)
                response = await copilot.chat(user_input)
                print(response + "\n")

            except KeyboardInterrupt:
                print("\n🤖 Investment Copilot: Goodbye! 👋")
                break
            except Exception as e:
                print(f"🤖 Investment Copilot: Sorry, I encountered an error: {e}\n")


if __name__ == "__main__":
    print("Choose mode:")
    print("1. Demo conversation")
    print("2. Interactive session")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(demo_copilot_interaction())
    else:
        asyncio.run(interactive_copilot_session())
