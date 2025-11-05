"""
Terminal Command Executor Service.

This file defines the `TerminalExecutor` class, which manages a persistent
shell state, including the current working directory (CWD). It is responsible
for safely executing shell commands, capturing their output, and handling
special cases like the 'cd' command to maintain CWD state.
"""

import os
import subprocess
from typing import Dict, Any

class TerminalExecutor:
    """
    Manages and executes shell commands in a stateful manner.

    This class maintains a current working directory (CWD) and can
    execute arbitrary shell commands, correctly handling `cd` to update
    its internal state.

    Attributes:
        cwd (str): The current working directory for command execution.
    """
    def __init__(self, start_dir: str = None):
        """
        Initializes the TerminalExecutor.

        Args:
            start_dir (str, optional): The directory to start in.
                Defaults to the user's home directory if None.
        """
        if start_dir:
            self.cwd = os.path.realpath(start_dir)
        else:
            self.cwd = os.path.expanduser("~")
        
        if not os.path.isdir(self.cwd):
            print(f"Warning: Start directory {self.cwd} not found. Defaulting to home.")
            self.cwd = os.path.expanduser("~")
            
        print(f"TerminalExecutor initialized at: {self.cwd}")

    def execute(self, command: str) -> Dict[str, Any]:
        """
        Executes a shell command.

        This method handles the 'cd' command as a special case to update
        the `self.cwd` state. All other commands are run in a subprocess
        within the current `self.cwd`.

        Args:
            command (str): The shell command string to execute.

        Returns:
            Dict[str, Any]: A dictionary containing the command's 'output'
                            (stdout and stderr) and the updated 'cwd'.
        """
        if not command:
            return {'output': 'Error: No command provided', 'cwd': self.cwd}

        command_str = command.strip()

        try:
            if command_str.startswith('cd '):
                new_dir = command_str.split(' ', 1)[1]
                if not new_dir:
                    new_dir = "~"
                
                new_dir = os.path.expanduser(new_dir)
                
                if not os.path.isabs(new_dir):
                    new_dir = os.path.join(self.cwd, new_dir)
                
                if os.path.isdir(new_dir):
                    self.cwd = os.path.realpath(new_dir)
                    output = f"Changed directory to: {self.cwd}"
                else:
                    output = f"Error: Directory not found: {new_dir}"
                
                return {'output': output, 'cwd': self.cwd}

            result = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                encoding='utf-8',
                errors='ignore'
            )
            
            output = (result.stdout or "") + (result.stderr or "")
            return {'output': output.strip(), 'cwd': self.cwd}

        except Exception as e:
            print(f"Error executing command '{command_str}': {e}")
            return {'output': f"Execution Error: {str(e)}", 'cwd': self.cwd}