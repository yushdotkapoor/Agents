# Modular Bug Fixer Agents

This project provides a modular approach to debugging Python code by splitting functionality into reusable agent components.

## Overview

The original `bug_fixer.py` has been refactored into three separate, reusable agent modules:

1. **WebSearchAgent** (`web_search_agent.py`) - Handles web research for error solutions
2. **PRAgent** (`pr_agent.py`) - Manages git operations and pull request creation
3. **CodeTestAgent** (`code_test_agent.py`) - Tests code snippets and validates fixes

## Agent Modules

### WebSearchAgent

Performs web searches to find solutions for programming errors.

```python
from web_search_agent import WebSearchAgent

# Initialize with custom number of results
web_agent = WebSearchAgent(max_results=5)

# Search for error solutions
results = web_agent.search_error_solutions("ModuleNotFoundError: No module named 'requests'")

# General search
search_results = web_agent.search_general("Python async await tutorial")
```

**Key Features:**
- Web search using DuckDuckGo
- Error-specific solution searching
- Configurable result count
- Error handling for failed searches

### PRAgent

Handles git operations and pull request creation workflows with LLM-generated commit messages.

```python
from pr_agent import PRAgent
from langchain.chat_models import ChatAnthropic

# Initialize PR agent with LLM for commit message generation
llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
pr_agent = PRAgent(repo_path=".", llm=llm)

# Check if in git repo
if pr_agent.is_git_repo:
    # Generate intelligent commit message
    commit_message = pr_agent.generate_commit_message(
        error_message="ModuleNotFoundError: No module named 'requests'",
        fix_description="Added requests to requirements.txt",
        files_changed=["requirements.txt"]
    )
    
    # Create new branch
    branch_name = pr_agent.create_branch("fix-bug-123")
    
    # Stage and commit changes with generated message
    pr_agent.stage_files()
    pr_agent.commit_changes(commit_message)
    
    # Push to remote
    pr_agent.push_branch(branch_name)
```

**Key Features:**
- Git repository detection
- Branch creation and management
- File staging and committing
- Remote pushing
- Interactive PR creation workflow
- **LLM-based commit message generation**
- Conventional commit format support
- Automatic fallback to default messages

### CodeTestAgent

Tests code snippets and validates suggested fixes.

```python
from code_test_agent import CodeTestAgent

# Initialize with custom timeout
test_agent = CodeTestAgent(timeout=15)

# Test a code snippet
stdout, stderr = test_agent.test_code_snippet("print('Hello, World!')")

# Extract and test code from LLM response
extracted_code, stdout, stderr = test_agent.test_llm_response(llm_response)

# Validate syntax without execution
is_valid, error = test_agent.validate_code_syntax("def hello(): print('world')")
```

**Key Features:**
- Safe code execution in temporary files
- Code extraction from LLM responses
- Syntax validation
- Configurable execution timeout
- Formatted test result output

## Main Bug Fixer

The refactored `bug_fixer.py` now uses these modular agents:

```python
from web_search_agent import WebSearchAgent
from pr_agent import PRAgent
from code_test_agent import CodeTestAgent

def main():
    # Initialize agents
    web_agent = WebSearchAgent(max_results=3)
    pr_agent = PRAgent()  # Uses default LLM for commit messages
    test_agent = CodeTestAgent(timeout=10)
    
    # Use agents in workflow
    web_results = web_agent.search_error_solutions(error_input)
    # ... LLM processing ...
    extracted_code, stdout, stderr = test_agent.test_llm_response(response)
    
    # Extract fix description for better commit messages
    fix_description = extract_fix_description(response)
    pr_agent.maybe_create_pr(error_input, fix_description)
```

## Usage Examples

### Running the Main Bug Fixer

```bash
python bug_fixer.py
```

### Using Individual Agents

```bash
# Run example usage
python example_usage.py

# Use web search only
python -c "
from web_search_agent import WebSearchAgent
agent = WebSearchAgent()
print(agent.search_error_solutions('ImportError: No module named numpy'))
"
```

### Creating Custom Workflows

```python
from web_search_agent import WebSearchAgent
from code_test_agent import CodeTestAgent
from pr_agent import PRAgent

# Custom debugging workflow with LLM commit messages
web_agent = WebSearchAgent(max_results=5)
test_agent = CodeTestAgent(timeout=10)
pr_agent = PRAgent()  # Uses default LLM

error = "TypeError: 'NoneType' object is not callable"
solutions = web_agent.search_error_solutions(error)

# Test potential fixes
test_code = "def safe_call(func): return func() if func else None"
stdout, stderr = test_agent.test_code_snippet(test_code)

# Generate commit message for the fix
commit_message = pr_agent.generate_commit_message(
    error_message=error,
    fix_description="Added null check before function call",
    files_changed=["utils.py"]
)
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (for Anthropic API):
```bash
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

## Benefits of Modular Design

1. **Reusability**: Each agent can be used independently in other projects
2. **Testability**: Individual components can be tested in isolation
3. **Maintainability**: Easier to update and extend specific functionality
4. **Flexibility**: Mix and match agents for different use cases
5. **Backward Compatibility**: Legacy functions are preserved for existing code

## File Structure

```
Agents/
├── bug_fixer.py          # Main refactored bug fixer
├── web_search_agent.py   # Web search functionality
├── pr_agent.py          # Git and PR operations with LLM commit messages
├── code_test_agent.py   # Code testing and validation
├── example_usage.py     # Usage examples
├── example_commit_generation.py  # LLM commit message examples
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## Contributing

When adding new features:

1. Create new agent modules for new functionality
2. Keep agents focused on single responsibilities
3. Maintain backward compatibility with legacy functions
4. Add examples to `example_usage.py`
5. Update this README with new agent documentation 