"""
Color utilities for console output.
Provides colored text output to replace emojis with meaningful colored messages.
"""

import os
from typing import Optional

# ANSI color codes
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def colored(text: str, color: str, bold: bool = False) -> str:
    """
    Return colored text if colors are supported, otherwise return plain text.
    
    Args:
        text: The text to colorize
        color: The color to use (from Colors class)
        bold: Whether to make the text bold
    
    Returns:
        Colored text string
    """
    # Check if colors are supported (not in a pipe, not in Windows cmd without colorama)
    if not _supports_color():
        return text
    
    result = text
    if bold:
        result = Colors.BOLD + result
    result = color + result + Colors.RESET
    return result

def _supports_color() -> bool:
    """Check if the current terminal supports colors."""
    # Check if we're in a pipe or redirect
    if not os.isatty(1):
        return False
    
    # Check for NO_COLOR environment variable
    if os.getenv('NO_COLOR'):
        return False
    
    # Check for TERM environment variable
    term = os.getenv('TERM', '').lower()
    if term in ('dumb', 'unknown'):
        return False
    
    return True

# Convenience functions for common colored outputs
def error(text: str) -> str:
    """Return text in red color for errors."""
    return colored(text, Colors.RED, bold=True)

def success(text: str) -> str:
    """Return text in green color for success messages."""
    return colored(text, Colors.GREEN, bold=True)

def warning(text: str) -> str:
    """Return text in yellow color for warnings."""
    return colored(text, Colors.YELLOW, bold=True)

def info(text: str) -> str:
    """Return text in blue color for informational messages."""
    return colored(text, Colors.BLUE)

def highlight(text: str) -> str:
    """Return text in cyan color for highlights."""
    return colored(text, Colors.CYAN)

def bold(text: str) -> str:
    """Return text in bold."""
    return colored(text, Colors.WHITE, bold=True) 