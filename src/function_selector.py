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

def get_valid_tokens(
    generated: str,
    function_names: list[str],
    vocab: dict[str, int]
) -> set[int]:
    """Find valid token IDs given what's been generated so far.

    Args:
        generated: The text generated so far.
        function_names: List of valid function names.
        vocab: Dictionary mapping token strings to IDs.

    Returns:
        Set of valid token IDs.
    """
    valid_ids = set()

    for name in function_names:
        if name.startswith(generated):
            next_part = name[len(generated):]
            for token_str, token_id in vocab.items():
                if next_part.startswith(token_str):
                    valid_ids.add(token_id)

    return valid_ids


def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    model: Small_LLM_Model,
    vocab: dict[str, int]
) -> FunctionDefinition:
    """Use LLM with constrained decoding to select the right function.

    Args:
        prompt: The user's natural language request.
        functions: List of available function definitions.
        model: The LLM model to use.
        vocab: Dictionary mapping token strings to IDs.

    Returns:
        The selected FunctionDefinition.
    """
    # Step 1: build prompt
    prompt_text = build_prompt(prompt, functions)

    # Step 2: encode to token IDs
    input_ids = model.encode(prompt_text)[0].tolist()

    # Step 3: get all valid function names
    function_names = [fn.name for fn in functions]

    # Step 4: generate function name token by token
    generated = ""

    while True:
        # get logits for next token
        logits = model.get_logits_from_input_ids(input_ids)

        # find valid next tokens
        valid_token_ids = get_valid_tokens(generated, function_names, vocab)

        # set invalid tokens to -inf
        for i in range(len(logits)):
            if i not in valid_token_ids:
                logits[i] = float("-inf")

        # pick highest scoring valid token
        next_token_id = logits.index(max(logits))

        # find what string this token represents
        next_token_str = [k for k, v in vocab.items() if v == next_token_id][0]

        # add to generated text
        generated += next_token_str
        input_ids.append(next_token_id)

        # check if we have a complete function name
        if generated in function_names:
            break

    # Step 5: find and return the matching function
    for fn in functions:
        if fn.name == generated:
            return fn

    # fallback: return first function
    return functions[0]