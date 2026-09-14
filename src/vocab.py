import json
from typing import Dict


def load_vocab(path: str) -> Dict[str, int]:
    """Load a vocabulary JSON file and convert token IDs to integers.

    Args:
        path: Path to the vocabulary JSON file.

    Returns:
        Dictionary mapping token string to token ID.
    """
    try:
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"Error: vocab file not found: {path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in vocab file: {e}")
        return {}

    return {k: int(v) for k, v in data.items()}
