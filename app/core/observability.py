import os

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from app.core.logging import logger


def langfuse_init():
  langfuse = Langfuse(
      public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
      secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
      host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
  )

  if langfuse.auth_check():
    logger.info("langfuse_auth_success")
  else:
    logger.error("langfuse_auth_failure")


def get_langfuse_callback_handler() -> CallbackHandler:
  """Create a Langfuse CallbackHandler for tracking LLM interactions.

  Returns:
      CallbackHandler: Configured Langfuse callback handler.
  """

  return CallbackHandler()

langfuse_callback_handler = get_langfuse_callback_handler()
