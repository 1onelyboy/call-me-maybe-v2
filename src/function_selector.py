from src.models import FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model
from src import visualizer


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
    system_prompt = "You are a function calling assistant.\n"
    system_prompt += "Respond with ONLY the function name, nothing else.\n"
    system_prompt += "Available functions:\n"

    for fn in functions:
        system_prompt += f"- {fn.name}: {fn.description}\n"

    system_prompt += f"\nUser request: {prompt}\n"
    system_prompt += (
        "\nWhich function should be called? "
        "Answer with only the function name:\n"
    )

    return system_prompt


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
    vocab: dict[str, int],
    max_iterations: int = 42
) -> FunctionDefinition:
    """Use LLM with constrained decoding to select the right function.

    Args:
        prompt: The user's natural language request.
        functions: List of available function definitions.
        model: The LLM model to use.
        vocab: Dictionary mapping token strings to IDs.
        max_iterations: Safety cap on generated tokens.

    Returns:
        The selected FunctionDefinition.

    Raises:
        ValueError: If the generated name matches no known function.
    """
    prompt_text = build_prompt(prompt, functions)
    generation_ids = model.encode(prompt_text)[0].tolist()
    function_names = [fn.name for fn in functions]
    generated = ""

    for _ in range(max_iterations):
        all_token_scores = model.get_logits_from_input_ids(generation_ids)

        allowed_token_ids = get_valid_tokens(generated, function_names, vocab)

        # nothing valid can follow - stop instead of looping forever
        if not allowed_token_ids:
            break

        for token_id in range(len(all_token_scores)):
            if token_id not in allowed_token_ids:
                all_token_scores[token_id] = float("-inf")

        best_token_id = all_token_scores.index(max(all_token_scores))

        best_token_str = [
            token_str
            for token_str, token_id in vocab.items()
            if token_id == best_token_id
        ][0]

        generated += best_token_str
        visualizer.show_step(
            "function", len(allowed_token_ids), best_token_str, generated
        )
        generation_ids.append(best_token_id)

        if generated in function_names:
            break

    for fn in functions:
        if fn.name == generated:
            return fn

    raise ValueError(
        f"Generated name '{generated}' matched no known function"
    )
