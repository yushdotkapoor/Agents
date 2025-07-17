from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import uuid
import os
from dotenv import load_dotenv

load_dotenv()


class MemorySearchChatbot:
    def __init__(self):
        self.memory = MemorySaver()
        
        # Get model based on environment variable
        llm_source = os.getenv("LLM_SOURCE", "anthropic").lower()
        
        if llm_source == "ollama":
            # Use Ollama with a default model (you can change this to your preferred model)
            self.model = init_chat_model("ollama:qwen2.5:32b", temperature=0.7)
            print(f"Using Ollama model: qwen2.5:32b")
        else:
            # Default to Anthropic
            self.model = init_chat_model("anthropic:claude-3-5-sonnet-latest")
            print(f"Using Anthropic model: claude-3-5-sonnet-latest")
        
        self.search = TavilySearch(max_results=2)
        self.tools = [self.search]
        self.agent_executor = create_react_agent(self.model, self.tools, checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # Store conversation history for direct model calls
        self.conversation_history = []
        
    def needs_search(self, query):
        """Determine if a query needs web search"""
        search_indicators = [
            "search", "latest", "recent", "current", "today", "news", "what's happening",
            "update", "2024", "2025", "now", "price", "stock", "weather", "events"
        ]
        return any(indicator in query.lower() for indicator in search_indicators)
    
    def stream_direct_response(self, message):
        """Stream response directly from model without tools"""
        # Add message to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Stream response directly from model
        stream = self.model.stream(self.conversation_history)
        
        response_content = ""
        for chunk in stream:
            if hasattr(chunk, 'content') and chunk.content:
                print(chunk.content, end="", flush=True)
                response_content += chunk.content
        
        # Add AI response to history
        self.conversation_history.append({"role": "assistant", "content": response_content})
        
    def chat(self, message):
        """Send a message to the agent and return the response"""
        input_message = {
            "role": "user",
            "content": message,
        }
        
        response = ""
        for step in self.agent_executor.stream(
            {"messages": [input_message]}, self.config, stream_mode="values"
        ):
            if step["messages"]:
                last_message = step["messages"][-1]
                if hasattr(last_message, 'content') and last_message.type == "ai":
                    response = last_message.content
        
        return response
    
    def run_cli(self):
        """Run the CLI chatbot interface"""
        print("Memory Search Chatbot")
        print("Type 'quit', 'exit', or 'bye' to end the conversation")
        print("Type 'clear' to start a new conversation thread")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! Have a great day!")
                    break
                
                # Check for clear command
                if user_input.lower() == 'clear':
                    self.config = {"configurable": {"thread_id": str(uuid.uuid4())}}
                    self.conversation_history = []  # Clear direct chat history too
                    print("\nStarted new conversation thread")
                    continue
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Get response from agent
                print("\nAssistant: ", end="", flush=True)
                
                # Determine if we need search or can stream directly
                if self.needs_search(user_input):
                    # Use agent executor for search-based queries
                    input_message = {
                        "role": "user",
                        "content": user_input,
                    }
                    
                    response_content = ""
                    for step in self.agent_executor.stream(
                        {"messages": [input_message]}, self.config, stream_mode="values"
                    ):
                        if step["messages"]:
                            last_message = step["messages"][-1]
                            if hasattr(last_message, 'content') and last_message.type == "ai":
                                if last_message.content != response_content:
                                    # Print new content
                                    new_content = last_message.content[len(response_content):]
                                    print(new_content, end="", flush=True)
                                    response_content = last_message.content
                    
                    if not response_content.endswith('\n'):
                        print()
                else:
                    # Stream directly from model for simple queries
                    self.stream_direct_response(user_input)
                    print()
                            
            except KeyboardInterrupt:
                print("\n\nGoodbye! Have a great day!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")


def main():
    chatbot = MemorySearchChatbot()
    chatbot.run_cli()


if __name__ == "__main__":
    main()