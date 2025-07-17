#!/usr/bin/env python3
"""
CLI tool for the Agent System.

This tool allows you to run different main agents. Currently available:
- bug-fixing: Main bug fixing agent that orchestrates sub-agents

Usage:
    agent                    # Interactive menu
    agent bug-fixing         # Run bug fixing agent
    agent --help             # Show help
    agent --list             # List available agents
    agent --install          # Install to /usr/local/bin
    agent --uninstall        # Remove from /usr/local/bin
"""

import sys
import os
import argparse
import subprocess
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the Agents directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Agents'))

load_dotenv()


# Agent registry - add new agents here
AGENTS = {
    'bug-fixing': {
        'name': 'Bug Fixing Agent',
        'description': 'Main agent that orchestrates web search, code testing, and PR creation',
        'module': 'Agents.bug_fixing_agent',
        'function': 'main'
    },
    'code-tour': {
        'name': 'Codebase Tour Guide',
        'description': 'Ask questions about any repository and get cited answers',
        'module': 'Agents.codebase_tour_guide',
        'function': 'main'
    },
    'refactor': {
        'name': 'Refactor Assistant',
        'description': 'Automatically propose and optionally apply code refactors based on Ruff lint issues',
        'module': 'Agents.refactor_assistant',
        'function': 'main'
    },
    'trade': {
        'name': 'Alpaca Trading Agent',
        'description': 'Interactive trading CLI using Alpaca-py for account info, quotes, and orders',
        'module': 'Agents.stock_trading_agent',
        'function': 'main'
    },
    'chartsniper': {
        'name': 'ChartSniper AI Agent',
        'description': 'AI-powered chart analysis and trading using OCR and vision AI',
        'module': 'Agents.chart_sniper_agent',
        'function': 'main'
    },
}


def run_agent(agent_key):
    """Run a specific agent."""
    if agent_key not in AGENTS:
        print(f"❌ Unknown agent: {agent_key}")
        print(f"Available agents: {', '.join(AGENTS.keys())}")
        return
    
    agent_info = AGENTS[agent_key]
    print(f"=== {agent_info['name']} ===")
    print(f"{agent_info['description']}\n")
    
    try:
        # Special handling for ChartSniper to avoid argument conflicts
        if agent_key == 'chartsniper':
            import sys
            # Temporarily clear sys.argv to avoid argument parsing conflicts
            original_argv = sys.argv.copy()
            sys.argv = [sys.argv[0]]  # Keep only the script name
            
            try:
                # Import and run ChartSniper
                module = __import__(agent_info['module'], fromlist=[agent_info['function']])
                agent_function = getattr(module, agent_info['function'])
                agent_function()
            finally:
                # Restore original argv
                sys.argv = original_argv
        else:
            # Standard agent execution
            module = __import__(agent_info['module'], fromlist=[agent_info['function']])
            agent_function = getattr(module, agent_info['function'])
            agent_function()
        
    except ImportError as e:
        print(f"❌ Failed to import agent module: {e}")
    except AttributeError as e:
        print(f"❌ Failed to find agent function: {e}")
    except Exception as e:
        print(f"❌ Error running agent: {e}")


def show_menu():
    """Display the main menu."""
    print("🤖 Agent System")
    print("=" * 40)
    print("Choose an agent to run:")
    
    for i, (key, agent) in enumerate(AGENTS.items(), 1):
        print(f"{i}. {agent['name']}")
        print(f"   {agent['description']}")
    
    print(f"{len(AGENTS) + 1}. Exit")
    print("=" * 40)


def interactive_mode():
    """Run in interactive mode with menu."""
    while True:
        show_menu()
        
        try:
            choice = input(f"\nEnter your choice (1-{len(AGENTS) + 1}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if choice_num == len(AGENTS) + 1:
                    print("Goodbye! 👋")
                    break
                elif 1 <= choice_num <= len(AGENTS):
                    agent_key = list(AGENTS.keys())[choice_num - 1]
                    run_agent(agent_key)
                else:
                    print(f"Invalid choice. Please enter 1-{len(AGENTS) + 1}.")
            else:
                print("Please enter a number.")
                
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        input("\nPress Enter to continue...")


def list_agents():
    """List all available agents."""
    print("Available agents:")
    print("=" * 40)
    
    for key, agent in AGENTS.items():
        print(f"• {key}")
        print(f"  {agent['name']}")
        print(f"  {agent['description']}")
        print()


def install_agent():
    """Install the agent CLI tool to /usr/local/bin."""
    try:
        # Get the current script path
        current_script = os.path.abspath(__file__)
        
        # Create the wrapper script content
        wrapper_content = f"""#!/bin/bash
# Wrapper script for agent CLI tool
python3 {current_script} "$@"
"""
        # Write the wrapper script to a temporary file
        temp_wrapper = "/tmp/agent_wrapper"
        with open(temp_wrapper, 'w') as f:
            f.write(wrapper_content)
        
        # Make it executable
        os.chmod(temp_wrapper, 0o755)
        
        # Copy to /usr/local/bin using sudo
        result = subprocess.run(['sudo', 'cp', temp_wrapper, '/usr/local/bin/agent'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Agent CLI tool installed successfully!")
            print("You can now run 'agent' from anywhere in your terminal.")
            print("Try: agent --help")
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
        
        # Clean up temporary file
        os.remove(temp_wrapper)
        return True       
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False


def uninstall_agent():
    """Move the agent CLI tool from /usr/local/bin."""
    try:
        # Check if the agent is installed
        if not os.path.exists('/usr/local/bin/agent'):
            print("❌ Agent CLI tool is not installed.")
            return False
        
        # Remove using sudo
        result = subprocess.run(['sudo', 'rm', '/usr/local/bin/agent'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Agent CLI tool uninstalled successfully!")
        else:
            print(f"❌ Uninstallation failed: {result.stderr}")
            return False
        
        return True       
    except Exception as e:
        print(f"❌ Uninstallation failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Agent System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agent run                    # Interactive menu
  agent run bug-fixing         # Run bug fixing agent
  agent --list                 # List available agents
  agent --install              # Install to /usr/local/bin
  agent --uninstall            # Remove from /usr/local/bin
  agent --help                 # Show this help
        """
    )
    
    parser.add_argument(
        'command',
        nargs='?',
        choices=['run'],
        help='Command to run (default: show help)'
    )
    
    parser.add_argument(
        'agent',
        nargs='?',
        choices=list(AGENTS.keys()),
        help='Agent to run (when using "run" command)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available agents'
    )
    
    parser.add_argument(
        '--install',
        action='store_true',
        help='Install agent CLI tool to /usr/local/bin'
    )
    
    parser.add_argument(
        '--uninstall',
        action='store_true',
        help='Remove agent CLI tool from /usr/local/bin'
    )
    
    args = parser.parse_args()
    
    if args.install:
        install_agent()
    elif args.uninstall:
        uninstall_agent()
    elif args.list:
        list_agents()
    elif args.command == 'run':
        if args.agent is None:
            # Interactive mode
            interactive_mode()
        else:
            # Run specific agent
            run_agent(args.agent)
    else:
        # Show help if no valid command provided
        parser.print_help()


if __name__ == "__main__":
    main() 