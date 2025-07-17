from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import sys
from dotenv import load_dotenv

# Import our modular agents
from helpers.web_search_agent import WebSearchAgent
from helpers.pr_agent import PRAgent
from helpers.code_test_agent import CodeTestAgent
import os

load_dotenv()


# Define prompt template
prompt = PromptTemplate(
    input_variables=["error_log", "web_results"],
    template="""
You are a helpful programming assistant.
Given the following Python error message or traceback, identify the root cause, explain it in simple terms, and suggest a fix. If relevant public fixes or discussions were found, incorporate them.

---

{error_log}

---

Relevant Web Results:
{web_results}

---

Return your answer in the following format:
1. **Root Cause**: Explain the core issue
2. **Explanation**: A short paragraph in plain English
3. **Suggested Fix**: Show corrected code snippet or describe how to fix
"""
)

def choose_model():
    if os.environ["LLM_SOURCE"] == "anthropic":
        return ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
    elif os.environ["LLM_SOURCE"] == "ollama":
        return OllamaLLM(model="gemma3:4b")
    else:
        raise ValueError("Invalid LLM source")



def main():
    # Initialize agents
    web_agent = WebSearchAgent(max_results=3)
    pr_agent = PRAgent()
    test_agent = CodeTestAgent(timeout=10)
    
    print("Paste your Python traceback or error message. Press Enter twice to submit:")
    lines = []
    while True:
        line = sys.stdin.readline()
        if line.strip() == "":
            break
        lines.append(line)

    error_input = "".join(lines)
    
    # Use web search agent
    web_results = web_agent.search_error_solutions(error_input)
    
    # Get LLM response
    llm = choose_model()
    chain = prompt | llm
    response = chain.invoke({"error_log": error_input, "web_results": web_results})

    print("\n===== Debugging Agent Response =====\n")
    print(response)

    # Use code test agent to test the response
    extracted_code, stdout, stderr = test_agent.test_llm_response(response)
    
    if extracted_code:
        print("\n" + test_agent.format_test_results(stdout, stderr))
    else:
        print("\nNo testable code block found.")

    # Extract fix description from response for better commit messages
    fix_description = ""
    if "**Suggested Fix**" in response:
        fix_section = response.split("**Suggested Fix**")[1]
        if "```" in fix_section:
            fix_description = fix_section.split("```")[0].strip()
        else:
            fix_description = fix_section.strip()
    
    # Use PR agent for git operations with error message and fix description
    pr_agent.maybe_create_pr(error_input, fix_description)


if __name__ == "__main__":
    main()
