import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.models import FunctionCall
from src.loader import load_functions_definition, load_test_prompts
from llm_sdk.llm_sdk import Small_LLM_Model
from src.vocab import load_vocab
from src.function_selector import select_function
from src.decoder import generate_arguments
from src import visualizer


def setup_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Function calling tool for LLMs"
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file"
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/output/function_calling_results.json",
        help="Path to write the results JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model identifier to load (default: Qwen/Qwen3-0.6B)."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show how each answer is built, token by token"

    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the function calling tool."""
    args = setup_arguments()

    if args.visualize:
        visualizer.enable()

    functions = load_functions_definition(args.functions_definition)
    prompts = load_test_prompts(args.input)

    if not functions:
        print("Error: no functions loaded")
        sys.exit(1)
    if not prompts:
        print("Error: no prompts loaded")
        sys.exit(1)

    print(f"Loaded {len(functions)} functions")
    print(f"Loaded {len(prompts)} prompts")

    print(f"Loading model {args.model}...")
    try:
        model = Small_LLM_Model(args.model)
    except Exception as e:
        print(f"Error: cannot load model: {e}", file=sys.stderr)
        sys.exit(1)

    vocab = load_vocab(model.get_path_to_vocab_file())
    if not vocab:
        print("Error: vocabulary could not be loaded", file=sys.stderr)
        sys.exit(1)
    print("Model loaded!")

    results: list[dict[str, Any]] = []

    for p in prompts:
        print(f"\nPrompt: {p.prompt}")
        try:
            selected = select_function(p.prompt, functions, model, vocab)
            print(f"Selected function: {selected.name}")
            parameters = generate_arguments(p.prompt, selected, model, vocab)
            print(f"Arguments: {parameters}")
            name = selected.name
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            name = ""
            parameters = {}

        call = FunctionCall(prompt=p.prompt, name=name, parameters=parameters)
        results.append(call.model_dump())

    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
    except OSError as e:
        print(f"Error: cannot write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except KeyboardInterrupt:
        print(" interrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
