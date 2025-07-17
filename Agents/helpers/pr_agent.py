import subprocess
import tempfile
import os
from typing import Optional, Tuple
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


class PRAgent:
    """
    A reusable agent for handling git operations and pull request creation.
    """
    
    def __init__(self, repo_path: str = ".", llm=None):
        """
        Initialize the PR agent.
        
        Args:
            repo_path: Path to the git repository
            llm: Language model for generating commit messages
        """
        self.repo_path = repo_path
        self.is_git_repo = self._check_if_git_repo()
        self.llm = llm or self._get_default_llm()
        self._setup_commit_message_chain()
    
    def _get_default_llm(self):
        """Get default LLM for commit message generation."""
        try:
            return ChatAnthropic(model="claude-3-opus-20240229", temperature=0)
        except:
            # Fallback to Ollama if Anthropic is not available
            try:
                return OllamaLLM(model="gemma3:4b")
            except:
                return None
    
    def _setup_commit_message_chain(self):
        """Setup the LLM chain for commit message generation."""
        if not self.llm:
            return
        
        commit_prompt = PromptTemplate(
            input_variables=["error_message", "fix_description", "files_changed"],
            template="""
You are a helpful programming assistant that generates concise, descriptive commit messages.

Given the following information about a bug fix:
- Error message: {error_message}
- Fix description: {fix_description}
- Files changed: {files_changed}

Generate a clear, concise commit message that follows conventional commit format.
The message should be:
1. Descriptive but concise (under 72 characters)
2. Use conventional commit format (e.g., "fix: resolve import error in auth module")
3. Focus on what was fixed, not how it was fixed
4. Use present tense

Examples:
- "fix: resolve ModuleNotFoundError in web_search_agent"
- "feat: add timeout configuration to code test agent"
- "refactor: extract web search functionality into separate module"

Return only the commit message, nothing else.
"""
        )
        
        self.commit_chain = commit_prompt | self.llm
    
    def generate_commit_message(self, error_message: str, fix_description: str, files_changed: list = None) -> str:
        """
        Generate a commit message using LLM.
        
        Args:
            error_message: The original error that was fixed
            fix_description: Description of the fix applied
            files_changed: List of files that were changed
            
        Returns:
            Generated commit message
        """
        if not self.llm or not hasattr(self, 'commit_chain'):
            # Fallback to default message
            return "Apply fix suggested by AI agent"
        
        try:
            files_str = ", ".join(files_changed) if files_changed else "unknown"
            response = self.commit_chain.invoke({
                "error_message": error_message,
                "fix_description": fix_description,
                "files_changed": files_str
            })
            
            # Clean up the response
            commit_message = response.strip()
            if len(commit_message) > 72:
                commit_message = commit_message[:69] + "..."
            
            return commit_message
        except Exception as e:
            print(f"Failed to generate commit message: {e}")
            return "Apply fix suggested by AI agent"
    
    def get_staged_files(self) -> list:
        """
        Get list of staged files.
        
        Returns:
            List of staged file paths
        """
        if not self.is_git_repo:
            return []
        
        try:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            return []
        except Exception:
            return []
    
    def get_modified_files(self) -> list:
        """
        Get list of modified files (not yet staged).
        
        Returns:
            List of modified file paths
        """
        if not self.is_git_repo:
            return []
        
        try:
            result = subprocess.run(
                ["git", "ls-files", "--modified"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return [f.strip() for f in result.stdout.split('\n') if f.strip()]
            return []
        except Exception:
            return []
    
    def _check_if_git_repo(self) -> bool:
        """
        Check if the current directory is a git repository.
        
        Returns:
            True if it's a git repo, False otherwise
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_current_branch(self) -> Optional[str]:
        """
        Get the current git branch name.
        
        Returns:
            Current branch name or None if not in a git repo
        """
        if not self.is_git_repo:
            return None
        
        try:
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except Exception:
            return None
    
    def create_branch(self, branch_name: Optional[str] = None) -> Optional[str]:
        """
        Create and checkout a new branch.
        
        Args:
            branch_name: Name for the new branch. If None, generates one.
            
        Returns:
            Name of the created branch or None if failed
        """
        if not self.is_git_repo:
            return None
        
        if branch_name is None:
            # Generate a unique branch name
            branch_name = "fix-" + next(tempfile._get_candidate_names())
        
        try:
            # Create and checkout new branch
            subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=self.repo_path,
                check=True
            )
            return branch_name
        except subprocess.CalledProcessError as e:
            print(f"Failed to create branch: {e}")
            return None
    
    def stage_files(self, file_paths: Optional[list] = None) -> bool:
        """
        Stage files for commit.
        
        Args:
            file_paths: List of file paths to stage. If None, stages all changes.
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_git_repo:
            return False
        
        try:
            if file_paths:
                for file_path in file_paths:
                    subprocess.run(
                        ["git", "add", file_path],
                        cwd=self.repo_path,
                        check=True
                    )
            else:
                subprocess.run(
                    ["git", "add", "."],
                    cwd=self.repo_path,
                    check=True
                )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to stage files: {e}")
            return False
    
    def commit_changes(self, message: str) -> bool:
        """
        Commit staged changes.
        
        Args:
            message: Commit message
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_git_repo:
            return False
        
        try:
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to commit changes: {e}")
            return False
    
    def push_branch(self, branch_name: str, remote: str = "origin") -> bool:
        """
        Push a branch to remote repository.
        
        Args:
            branch_name: Name of the branch to push
            remote: Remote repository name
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_git_repo:
            return False
        
        try:
            subprocess.run(
                ["git", "push", "--set-upstream", remote, branch_name],
                cwd=self.repo_path,
                check=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to push branch: {e}")
            return False
    
    def create_fix_pr(self, error_message: str, fix_description: str, commit_message: str = None) -> Optional[str]:
        """
        Create a complete fix workflow: new branch, stage, commit, and push.
        
        Args:
            error_message: The original error that was fixed
            fix_description: Description of the fix applied
            commit_message: Custom commit message. If None, generates one using LLM.
            
        Returns:
            Name of the created branch or None if failed
        """
        if not self.is_git_repo:
            print("Not in a git repository.")
            return None
        
        # Create new branch
        branch_name = self.create_branch()
        if not branch_name:
            return None
        
        # Stage all changes
        if not self.stage_files():
            return None
        
        # Generate or use provided commit message
        if commit_message is None:
            files_changed = self.get_staged_files()
            commit_message = self.generate_commit_message(error_message, fix_description, files_changed)
        
        # Commit changes
        if not self.commit_changes(commit_message):
            return None
        
        # Push to remote
        if not self.push_branch(branch_name):
            return None
        
        return branch_name
    
    def maybe_create_pr(self, error_message: str = "", fix_description: str = "") -> Optional[str]:
        """
        Interactive PR creation - asks user if they want to create a PR.
        
        Args:
            error_message: The original error that was fixed
            fix_description: Description of the fix applied
            
        Returns:
            Name of the created branch or None if user declined or failed
        """
        if not self.is_git_repo:
            return None
        
        choice = input("This is a git repo. Create a new branch and PR for the fix? (y/n): ").strip().lower()
        if choice == 'y':
            branch_name = self.create_fix_pr(error_message, fix_description)
            if branch_name:
                print(f"\nPushed fix to branch '{branch_name}'. Create a PR on GitHub or your git server.")
            return branch_name
        return None

