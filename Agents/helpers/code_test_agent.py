import subprocess
import tempfile
import os
import re
from typing import Tuple, Optional


class CodeTestAgent:
    """
    A reusable agent for testing code snippets and suggested fixes.
    """
    
    def __init__(self, timeout: int = 10):
        """
        Initialize the code test agent.
        
        Args:
            timeout: Timeout in seconds for code execution
        """
        self.timeout = timeout
    
    def test_code_snippet(self, code: str, file_extension: str = ".py") -> Tuple[str, str]:
        """
        Test a code snippet by writing it to a temporary file and executing it.
        
        Args:
            code: The code snippet to test
            file_extension: File extension for the temporary file
            
        Returns:
            Tuple of (stdout, stderr)
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension, delete=False) as temp:
            temp.write(code)
            temp_path = temp.name
        
        try:
            result = subprocess.run(
                ["python3", temp_path], 
                capture_output=True, 
                text=True, 
                timeout=self.timeout
            )
            os.remove(temp_path)
            return result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            os.remove(temp_path)
            return "", f"Code execution timed out after {self.timeout} seconds"
        except Exception as e:
            try:
                os.remove(temp_path)
            except:
                pass
            return "", str(e)
    
    def extract_code_from_response(self, response: str) -> Optional[str]:
        """
        Extract code blocks from LLM response.
        
        Args:
            response: The LLM response containing code blocks
            
        Returns:
            Extracted code or None if no code block found
        """
        # Look for Python code blocks
        python_pattern = r"```python\s*\n(.*?)\n```"
        python_match = re.search(python_pattern, response, re.DOTALL)
        if python_match:
            return python_match.group(1).strip()
        
        # Look for generic code blocks
        code_pattern = r"```\s*\n(.*?)\n```"
        code_match = re.search(code_pattern, response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Look for inline code blocks
        inline_pattern = r"```(.*?)```"
        inline_match = re.search(inline_pattern, response, re.DOTALL)
        if inline_match:
            return inline_match.group(1).strip()
        
        return None
    
    def test_llm_response(self, response: str) -> Tuple[Optional[str], str, str]:
        """
        Test code extracted from an LLM response.
        
        Args:
            response: The LLM response containing code
            
        Returns:
            Tuple of (extracted_code, stdout, stderr)
        """
        extracted_code = self.extract_code_from_response(response)
        if extracted_code:
            stdout, stderr = self.test_code_snippet(extracted_code)
            return extracted_code, stdout, stderr
        else:
            return None, "", "No testable code block found."
    
    def validate_code_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Validate Python code syntax without executing it.
        
        Args:
            code: The code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Compilation error: {e}"
    
    def format_test_results(self, stdout: str, stderr: str) -> str:
        """
        Format test results for display.
        
        Args:
            stdout: Standard output from code execution
            stderr: Standard error from code execution
            
        Returns:
            Formatted test results string
        """
        result = "===== Test Run Output =====\n"
        
        if stdout.strip():
            result += f"STDOUT:\n{stdout}\n"
        else:
            result += "STDOUT: (empty)\n"
        
        if stderr.strip():
            result += f"STDERR:\n{stderr}\n"
        else:
            result += "STDERR: (empty)\n"
        
        return result


# Convenience function for backward compatibility
def test_fix(code: str) -> Tuple[str, str]:
    """
    Legacy function for testing code fixes.
    
    Args:
        code: The code to test
        
    Returns:
        Tuple of (stdout, stderr)
    """
    agent = CodeTestAgent()
    return agent.test_code_snippet(code) 