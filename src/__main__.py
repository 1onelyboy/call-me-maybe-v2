import sys
import argparse
from pathlib import Path
from src.loader import load_functions_definition, load_test_prompts
from llm_sdk.llm_sdk import Small_LLM_Model
from src.vocab import load_vocab
from src.function_selector import select_function
from src.decoder import generate_arguments


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
    return parser.parse_args()


def main() -> None:
    """Main entry point for the function calling tool."""
    args = setup_arguments()

    functions = load_functions_definition(args.functions_definition)
    prompts = load_test_prompts(args.input)

    if not functions:
        print("Error: no functions loaded")
        sys.exit(1)
    if not prompts:
        print("Error: no prompts loaded")
        sys.exit(1)

    print(f"Loaded {len(functions)} functions")
    # print(functions)
    print(f"Loaded {len(prompts)} prompts")
    # print(prompts)

    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    vocab = load_vocab(vocab_path)

    for p in prompts:
        print(f"\nPrompt: {p.prompt}")
        try:
            selected = select_function(p.prompt, functions, model, vocab)
            print(f"Selected function: {selected.name}")
        except ValueError as e:
            print(f"Error: {e}")
            continue
        args = generate_arguments(p.prompt, selected, model, vocab)
        print(f"Arguments: {args}")

    # # inside main(), after loading functions and prompts
    # for p in prompts:
    #     prompt_text = build_prompt(p.prompt, functions)
    #     print(prompt_text)
    #     print("---")
    #     break  # just test first prompt for now


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
