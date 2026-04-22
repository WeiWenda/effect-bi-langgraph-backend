"""Skill execution tools for running bash commands."""

import json
import subprocess
from langchain_core.tools import tool

from app.core.logging import logger


@tool
async def skill_exec_bash_cmd(command: str) -> str:
    """Execute a bash command that may be mentioned in skill documentation.

    Use this tool to run shell commands that are required for skill setup,
    configuration, or execution as described in skill.md files.

    Args:
        command: The bash command to execute (e.g., "echo 'Hello World'" or "ls -la").

    Returns:
        str: JSON string containing command output, exit code, and any errors.
    """
    try:
        logger.info("executing_bash_command", command=command)
        
        # Execute the command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        success = result.returncode == 0
        
        if success:
            logger.info("bash_command_success", command=command, exit_code=result.returncode)
            return json.dumps({
                "success": True,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "message": f"Command executed successfully with exit code {result.returncode}"
            })
        else:
            logger.warning("bash_command_failed", command=command, exit_code=result.returncode)
            return json.dumps({
                "success": False,
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "message": f"Command failed with exit code {result.returncode}"
            })
            
    except subprocess.TimeoutExpired:
        logger.error("bash_command_timeout", command=command)
        return json.dumps({
            "success": False,
            "command": command,
            "error": "Command timed out after 30 seconds",
            "message": "Command execution timed out"
        })
    except Exception as e:
        logger.exception("bash_command_exception", command=command, error=str(e))
        return json.dumps({
            "success": False,
            "command": command,
            "error": str(e),
            "message": f"Failed to execute command: {str(e)}"
        })
