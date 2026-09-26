"""Optional display of the constrained decoding, step by step.

Enabled with --visualize.
"""

_state = {"enabled": False}


def enable() -> None:
    """Turn the display on."""
    _state["enabled"] = True


def show_step(label: str, allowed: int, chosen: str, generated: str) -> None:
    """Print one decoding step if the display is on.

    Args:
        label: What is being generated.
        allowed: How many tokens were allowed at this step.
        chosen: The token the model picked.
        generated: The text generated so far.
    """
    if _state["enabled"]:
        token = chosen.replace("\u0120", " ")
        print(f"    {label}: {allowed} allowed, chose {token!r}"
              f" -> {generated!r}")