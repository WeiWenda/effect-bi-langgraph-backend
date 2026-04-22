"""Environment variable checking tool."""

import json
import os
from langchain_core.tools import tool

from app.core.logging import logger


@tool
async def check_env_key(key: str) -> str:
    """Check if an environment variable exists in the current environment.

    Use this tool to verify if required environment variables are configured
    before attempting to use skills or services that depend on them.

    Args:
        key: The environment variable name to check (e.g., "BAIDU_API_KEY").

    Returns:
        str: JSON string indicating whether the environment variable exists and is non-empty.
    """
    try:
        logger.info("checking_env_key", key=key)
        
        value = os.getenv(key)
        exists = value is not None and value.strip() != ""
        
        if exists:
            logger.info("env_key_exists", key=key)
            return json.dumps({
                "success": True,
                "key": key,
                "exists": True,
                "message": f"Environment variable '{key}' is configured"
            })
        else:
            logger.warning("env_key_not_found", key=key)
            return json.dumps({
                "success": True,
                "key": key,
                "exists": False,
                "message": f"Environment variable '{key}' is not configured or is empty"
            })
    except Exception as e:
        logger.exception("check_env_key_failed", key=key, error=str(e))
        return json.dumps({"error": f"Failed to check environment variable '{key}': {str(e)}"})
