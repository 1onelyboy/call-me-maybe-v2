import json
from typing import Any
from pydantic import ValidationError
from src.models import TestPrompt, FunctionDefinition


def load_json_file(path: str) -> Any:
    """Load and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as Python object.
    """
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: input file not found: {path}")
    except IsADirectoryError:
        print(f"Error: path is a directory, not a file: {path}")
    except PermissionError:
        print(f"Error: permission denied: {path}")
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in {path}: {e}")
    return None


def load_functions_definition(path: str) -> list[FunctionDefinition]:
    """Load function definitions from JSON file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        List of function definitions.
    """
    data = load_json_file(path)
    if data is None: 
        return []
    if not isinstance(data, list):
        print("Error: functions definition must be a JSON array")
        return []

    functions = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Error: function #{i} is not a dict")
            return []
        try:
            functions.append(FunctionDefinition(**item))
        except ValidationError as e:
            print(f"Error: function #{i} is invalid: {e}")
            return []

    return functions


def load_test_prompts(path: str) -> list[TestPrompt]:
    """Load test prompts from JSON file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        List of prompt objects.
    """
    data = load_json_file(path)
    if data is None:
        return []
    if not isinstance(data, list):
        print("Error: test prompts must be a JSON array")
        return []

    prompts = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"Error: test prompt[{i}] must be a dict")
            return []
        try:
            prompts.append(TestPrompt(**item))
        except ValidationError as e:
            print(f"Error: prompt #{i} is invalid: {e}")
            return []

    return prompts