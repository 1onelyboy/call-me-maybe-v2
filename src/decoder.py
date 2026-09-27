from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
import string


def get_string_tokens(vocab: dict[str, int]) -> set[int]:
    """Get token IDs that are valid inside a JSON string value.

    Args:
        vocab: Dictionary mapping token strings to IDs.

    Returns:
        Set of token IDs made only of safe characters.
    """
    safe_punctuation = string.punctuation.replace('"', "")
    allowed_chars = set(
        string.ascii_letters + string.digits + " Ġ" + safe_punctuation
    )
    valid_ids = set()

    for token_str, token_id in vocab.items():
        if token_str and all(char in allowed_chars for char in token_str):
            valid_ids.add(token_id)

    return valid_ids


def get_number_tokens(vocab: dict[str, int]) -> set[int]:
    """Get token IDs that are valid inside a JSON number value.

    Args:
        vocab: Dictionary mapping token strings  to IDs.

    Returns:
        Set of token IDs that are digits, minus sign or decimal point.
    """
    allowed_chars = set("0123456789.-")
    valid_ids = set()

    for token_str, token_id in vocab.items():
        if token_str and all(char in allowed_chars for char in token_str):
            valid_ids.add(token_id)

    return valid_ids


def token_id_to_text(token_id: int, vocab: dict[str, int]) -> str:
    """Convert a token ID back to its text.

    Args:
        token_id: The token ID to convert.
        vocab: Dictionary mapping token strings to IDs.

    Returns:
        The token's text, or empty string if not found.
    """
    for token_str, tid in vocab.items():
        if tid == token_id:
            return token_str
    return ""


def pick_best_token(scores: list[float], allowed_ids: set[int]) -> int:
    """Pick the allowed token with the highest score.

    Args:
        scores: List of scores for every token.
        allowed_ids: Set of token IDs that are allowed.

    Returns:
        The token ID with the highest score, or -1 if none allowed.
    """
    best_id = -1
    best_score = float("-inf")

    for token_id in allowed_ids:
        if token_id < len(scores) and scores[token_id] > best_score:
            best_score = scores[token_id]
            best_id = token_id

    return best_id


def get_closing_tokens(vocab: dict[str, int]) -> set[int]:
    """Get the tokens that can end a string value.

    Args:
        vocab: Dictionary mapping token strings to IDs.

    Returns:
        Set of token IDs that close a string.
    """
    letters_and_digits = set(string.ascii_letters + string.digits)
    closing_ids = set()

    for token_str, token_id in vocab.items():
        if token_str.startswith('"'):
            rest = token_str[1:]
            if not any(char in letters_and_digits for char in rest):
                closing_ids.add(token_id)

    return closing_ids


def generate_string_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[str, int],
    max_tokens: int = 50
) -> str:
    """Generate a string value, stopping at the closing quote.

    Args:
        model: The LLM model to use.
        input_ids: Current token IDs including the opening quote.
        vocab: Dictionary mapping token strings to IDs.
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated string value without quotes.
    """
    string_tokens = get_string_tokens(vocab)
    closing_ids = get_closing_tokens(vocab)
    allowed_ids = string_tokens | closing_ids

    value = ""

    for _ in range(max_tokens):
        scores = model.get_logits_from_input_ids(input_ids)
        best_id = pick_best_token(scores, allowed_ids)

        if best_id == -1 or best_id in closing_ids:
            break

        value += token_id_to_text(best_id, vocab)
        input_ids.append(best_id)

    return value.replace("\u0120", " ").strip()


def generate_number_value(
    model: Small_LLM_Model,
    input_ids: list[int],
    vocab: dict[str, int],
    max_tokens: int = 10
) -> float:
    """Generate a number value using constrained decoding.

    Args:
        model: The LLM model to use.
        input_ids: Current token IDs.
        vocab: Dictionary mapping token strings to IDs.
        max_tokens: Maximum tokens to generate.

    Returns:
        The generated number as a float.
    """
    number_tokens = get_number_tokens(vocab)
    value = ""

    for _ in range(max_tokens):
        scores = model.get_logits_from_input_ids(input_ids)
        best_id = pick_best_token(scores, number_tokens)

        if best_id == -1:
            break

        token_text = token_id_to_text(best_id, vocab)
        candidate = value + token_text

        try:
            float(candidate)
        except ValueError:
            break

        value = candidate
        input_ids.append(best_id)

    try:
        return float(value)
    except ValueError:
        return 0.0


def build_argument_prompt(
    prompt: str,
    function: FunctionDefinition
) -> str:
    """Build a prompt asking the LLM to fill in function arguments.

    Args:
        prompt: The user's natural language request.
        function: The selected function definition.

    Returns:
        A formatted prompt string.
    """
    lines = [
        "Extract function arguments. Copy values exactly.",
        "",
        "Request: Say hello to alice",
        'Arguments: {"name": "alice"}',
        "",
        "Request: What is 7 times 12?",
        'Arguments: {"a": 7, "b": 12}',
        "",
        "Request: Replace all vowels in 'hi there' with stars",
        'Arguments: {"source_string": "hi there", '
        '"regex": "[aeiouAEIOU]", "replacement": "*"}',
        "",
        f"Function: {function.name} - {function.description}",
        f"Request: {prompt}",
        "Arguments:",
    ]
    return "\n".join(lines)


def generate_arguments(
    prompt: str,
    function: FunctionDefinition,
    model: Small_LLM_Model,
    vocab: dict[str, int]
) -> dict[str, object]:
    """Generate arguments for a function call using constrained decoding.

    Args:
        prompt: The user's natural language request.
        function: The selected function definition.
        model: The LLM model to use.
        vocab: Dictionary mapping token strings to IDs.

    Returns:
        Dictionary mapping argument names to their values.
    """
    prompt_text = build_argument_prompt(prompt, function)
    input_ids = model.encode(prompt_text)[0].tolist()

    result: dict[str, object] = {}
    param_names = list(function.parameters.keys())

    json_prefix = "{"
    input_ids += model.encode(json_prefix)[0].tolist()

    for i, param_name in enumerate(param_names):
        param_type = function.parameters[param_name].type

        if i > 0:
            key_part = f', "{param_name}": '
        else:
            key_part = f'"{param_name}": '

        input_ids += model.encode(key_part)[0].tolist()

        if param_type == "string":
            input_ids += model.encode('"')[0].tolist()
            value = generate_string_value(model, input_ids, vocab)
            result[param_name] = value
            input_ids += model.encode('"')[0].tolist()
        else:
            number = generate_number_value(model, input_ids, vocab)
            result[param_name] = number
    return result
