from ddgs import DDGS
from typing import List, Dict, Optional


class WebSearchAgent:
    """
    A reusable agent for performing web searches to find relevant information
    about programming errors, bugs, and solutions.
    """
    
    def __init__(self, max_results: int = 3):
        """
        Initialize the web search agent.
        
        Args:
            max_results: Maximum number of search results to return
        """
        self.max_results = max_results
    
    def search_error_solutions(self, error_input: str) -> str:
        """
        Search for solutions to programming errors.
        
        Args:
            error_input: The error message or traceback to search for
            
        Returns:
            Formatted string of search results
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(error_input, max_results=self.max_results)
                if results:
                    formatted_results = []
                    for r in results:
                        formatted_results.append(f"- {r['title']}: {r['href']}")
                    return "\n".join(formatted_results)
                else:
                    return "No relevant results found."
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def search_general(self, query: str) -> List[Dict]:
        """
        Perform a general web search.
        
        Args:
            query: Search query
            
        Returns:
            List of search result dictionaries
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=self.max_results)
                return results if results else []
        except Exception as e:
            print(f"Search failed: {str(e)}")
            return []
    
    def format_results_for_prompt(self, results: List[Dict]) -> str:
        """
        Format search results for inclusion in LLM prompts.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Formatted string of results
        """
        if not results:
            return "No relevant results found."
        
        formatted_results = []
        for r in results:
            formatted_results.append(f"- {r['title']}: {r['href']}")
        return "\n".join(formatted_results)


# Convenience function for backward compatibility
def web_research(error_input: str, max_results: int = 3) -> str:
    """
    Legacy function for web research.
    
    Args:
        error_input: The error message to search for
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results
    """
    agent = WebSearchAgent(max_results=max_results)
    return agent.search_error_solutions(error_input) 