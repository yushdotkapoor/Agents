# AI Trading and Development Agents

A comprehensive collection of AI-powered agents for automated trading, code analysis, and development workflows.

## 🚀 Main Agents

### 1. ChartSniper Agent
**Location**: `Agents/chart_sniper_agent.py`

An advanced AI-powered trading agent that analyzes stock charts using computer vision and executes trades automatically.

**Features:**
- 📊 **Chart Analysis**: OCR-powered text extraction from stock charts
- 🤖 **AI Trading Decisions**: LLM-powered analysis using Claude or Ollama
- 📈 **Real-time Data**: Integration with Alpaca API for live market data
- 🎯 **Smart Execution**: Automated trade placement with risk management
- 📱 **Interactive CLI**: Clean, persistent command-line interface
- 📝 **Trade Logging**: Comprehensive history tracking and analysis
- 🎨 **Visual Charts**: Generate and analyze stock charts automatically

**Usage:**
```bash
# Interactive mode
python Agents/chart_sniper_agent.py

# Analyze a specific symbol
python Agents/chart_sniper_agent.py --symbol AAPL

# Analyze chart image
python Agents/chart_sniper_agent.py chart_image.png
```

### 2. Stock Trading Agent
**Location**: `Agents/stock_trading_agent.py`

A straightforward trading agent with manual controls for executing trades through the Alpaca API.

**Features:**
- 💰 **Account Management**: View account balance, equity, and buying power
- 💹 **Real-time Quotes**: Get current prices and market data
- 📊 **Position Tracking**: Monitor open positions and portfolio
- 🔄 **Trade Execution**: Buy/sell orders with market pricing
- 📱 **Interactive CLI**: Simple command-line trading interface

**Usage:**
```bash
python Agents/stock_trading_agent.py
```

### 3. Bug Fixing Agent
**Location**: `Agents/bug_fixing_agent.py`

An intelligent debugging assistant that helps identify and fix code issues automatically.

**Features:**
- 🔍 **Error Analysis**: Automated error detection and diagnosis
- 🌐 **Web Research**: Searches for solutions online using DuckDuckGo
- 🧪 **Code Testing**: Validates fixes before applying them
- 📋 **PR Creation**: Automatically creates pull requests with fixes
- 🤖 **LLM Integration**: Uses AI to understand and solve complex issues

**Usage:**
```bash
python Agents/bug_fixing_agent.py
```

### 4. Codebase Tour Guide
**Location**: `Agents/codebase_tour_guide.py`

An AI assistant that helps developers understand and navigate large codebases efficiently.

**Features:**
- 🗂️ **Code Indexing**: Creates searchable embeddings of your codebase
- 🔍 **Smart Search**: Find relevant code sections using natural language
- 📚 **Documentation**: Explains code structure and functionality
- 🧭 **Navigation**: Helps locate specific functions, classes, or patterns
- 💡 **Insights**: Provides architectural understanding and suggestions

**Usage:**
```bash
python Agents/codebase_tour_guide.py
```

### 5. Refactor Assistant
**Location**: `Agents/refactor_assistant.py`

An automated code refactoring agent that improves code quality and maintainability.

**Features:**
- 🔧 **Lint Analysis**: Uses Ruff to identify code quality issues
- ✨ **Auto-fixing**: Automatically applies fixes for common problems
- 🧪 **Test Validation**: Ensures refactored code still passes tests
- 📋 **PR Integration**: Creates pull requests with refactoring changes
- 📊 **Quality Metrics**: Tracks improvements in code quality

**Usage:**
```bash
python Agents/refactor_assistant.py
```

### 6. Memory Search Chatbot
**Location**: `Agents/memory_search_agent.py`

An intelligent CLI chatbot with web search capabilities and persistent conversation memory.

**Features:**
- 💬 **Interactive Chat**: Real-time streaming conversational interface
- 🔍 **Web Search**: Automatic web search integration via Tavily API
- 🧠 **Memory**: Persistent conversation history within sessions
- 🚀 **Smart Streaming**: Token-by-token response streaming for simple queries
- 🛠️ **Multi-LLM Support**: Choose between Anthropic Claude and Ollama
- 🎯 **Intelligent Routing**: Automatically determines when to use search vs. direct response

**Usage:**
```bash
# Through the agent CLI
python agent.py run memory-search

# Direct execution
python Agents/memory_search_agent.py
```

**Commands:**
- `quit`, `exit`, `bye` - End conversation
- `clear` - Start new conversation thread
- Any query - Get AI response with optional web search

## 🛠️ Helper Modules

Located in `Agents/helpers/`:
- `web_search_agent.py` - Web search capabilities
- `pr_agent.py` - Git operations and PR management
- `code_test_agent.py` - Code testing and validation
- `conditional_trading_agent.py` - Advanced trading conditions
- `symbol_fetcher.py` - Stock symbol management

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Trading APIs
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_PAPER=true  # Set to false for live trading

# LLM Configuration
LLM_SOURCE=anthropic  # or 'ollama'
ANTHROPIC_API_KEY=your_anthropic_api_key
OLLAMA_MODEL=llama3.2  # if using Ollama

# Web Search (for Memory Search Chatbot)
TAVILY_API_KEY=your_tavily_api_key

# Optional: Debug mode
CHARTSNIPER_DEBUG=true
```

### API Setup

1. **Alpaca Trading**: Sign up at [alpaca.markets](https://alpaca.markets) for trading API access
2. **Anthropic Claude**: Get API key from [console.anthropic.com](https://console.anthropic.com)
3. **Tavily Search**: Get API key from [tavily.com](https://tavily.com) for web search capabilities
4. **Ollama** (Alternative): Install locally from [ollama.ai](https://ollama.ai)

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yushdotkapoor/Agents.git
   cd Agents
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (create `.env` file)

4. **Run an agent**:
   ```bash
   # Start the ChartSniper trading agent
   python Agents/chart_sniper_agent.py
   
   # Or try the codebase tour guide
   python Agents/codebase_tour_guide.py
   ```

## 📊 Trading Features

- **Paper Trading**: Test strategies without real money
- **Risk Management**: Built-in stop-loss and take-profit features
- **Multiple LLM Support**: Choose between Claude and Ollama
- **Visual Analysis**: Chart generation and pattern recognition
- **Trade History**: Comprehensive logging and analysis

## 🛡️ Security

- All API keys are stored in environment variables
- No sensitive data is committed to the repository
- Paper trading mode available for safe testing
- Trade confirmation prompts for safety

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading involves financial risk, and past performance does not guarantee future results. Use at your own risk and never invest more than you can afford to lose.

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