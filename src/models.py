from typing import Any
from pydantic import BaseModel


class Parameter(BaseModel):
    """Represents a function parameter or return type.

    Attributes:
        type: The type of the parameter (number, string, boolean).
    """

    type: str


class FunctionDefinition(BaseModel):
    """Represents a function definition from functions_definition.json.

    Attributes:
        name: The function name.
        description: What the function does.
        parameters: Dict of parameter names to their types.
        returns: The return type of the function.
    """

    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter


class TestPrompt(BaseModel):
    """Represents a test prompt from function_calling_tests.json.

    Attributes:
        prompt: The natural language request.
    """

    prompt: str


class FunctionCall(BaseModel):
    """Represents a function call result for the output file.

    Attributes:
        prompt: The original natural language request.
        name: The function name to call.
        parameters: The arguments to pass to the function.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]
