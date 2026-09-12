from src.models import FunctionDefinition
from llm_sdk import Small_LLM_Model


def build_prompt(
    prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    """Build a prompt for the LLM to select the right function.

    Args:
        prompt: The user's natural language request.
        functions: List of available function definitions.

    Returns:
        A formatted prompt string.
    """
    text = "You are a function calling assistant.\n"
    text += "Available functions:\n"

    for fn in functions:
        text += f"- {fn.name}: {fn.description}\n"
    
    text+= f"\nUser request: {prompt}\n"
    text += "\nWhich function should be called? Answer with only the function name:\n"
    
    return text