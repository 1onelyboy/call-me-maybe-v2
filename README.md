*This project has been created as part of the 42 curriculum by trohain.*

# call me maybe

## Description

This project turns natural language prompts into function calls. It
uses the small model `Qwen/Qwen3-0.6B`.

```
"What is the sum of 2 and 3?"  ->  {"name": "fn_add_numbers",
                                    "parameters": {"a": 2.0, "b": 3.0}}
```

The program does not answer the question. It picks one of the
functions listed in `functions_definition.json` and extracts its
arguments with the right types. A small model is not reliable when it
writes JSON on its own, so the output is built with constrained
decoding: at each step the invalid tokens are removed before the model
can choose.

The pipeline is short. `src/loader.py` reads and validates the input
files, `src/function_selector.py` chooses the function name,
`src/decoder.py` produces the arguments, and `src/__main__.py` writes
one result per prompt to the output file.

## Instructions

Requirements: Python 3.10 or newer, and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # install the dependencies
uv run python -m src     # run with the default paths
```

The first run downloads the model into the Hugging Face cache. On
Linux, `pyproject.toml` uses the CPU-only index for torch, so the CUDA
packages are not downloaded.

The Makefile wraps the same commands:

| Rule | What it does |
|---|---|
| `make install` | install the dependencies (`uv sync`) |
| `make run` | run the program |
| `make visualize` | run the program with `--visualize` |
| `make debug` | run the program under `pdb` |
| `make clean` | remove `__pycache__` and `.mypy_cache` |
| `make lint` | run `flake8` and `mypy` |

## Example usage

```bash
# default paths
uv run python -m src

# custom paths
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json

# show the decoding step by step (bonus)
uv run python -m src --visualize

# list all the options
uv run python -m src --help
```

The input prompts look like this:

```json
[{"prompt": "What is the sum of 2 and 3?"}]
```

The output file contains one entry per prompt, in the same order:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Algorithm explanation

Text is generated one token at a time. At each step the model returns
a score, called a logit, for every token of its vocabulary. Normally
the token with the best score is chosen, and nothing stops the model
from choosing a token that breaks the format.

Constrained decoding changes this. Before the choice, every token that
is not valid at this point is set to negative infinity. Such a token
can never have the best score, so only a valid token can be chosen.
The vocabulary file returned by `get_path_to_vocab_file()` is loaded by
`src/vocab.py`. It maps each token to its id, which is how the program
knows what a token id means.

### Choosing the function

`src/function_selector.py` builds a prompt that lists the available
functions and asks for a name. The name is then generated token by
token. A token is allowed only if the text generated so far, plus that
token, is still the beginning of one of the real function names.

```
generated ""         -> allowed: "fn", "f", "fn_" ...
generated "fn_"      -> allowed: "add", "greet", "reverse", "get" ...
generated "fn_greet" -> complete name, stop
```

The loop stops as soon as the generated text is a full function name,
or when no token is allowed any more. The result can only be one of
the defined names. The model still decides which one, but it cannot
invent a name that does not exist.

### Generating the arguments

`src/decoder.py` does not ask the model for JSON. The program writes
the structure itself: the braces, the keys, the quotes, the colons and
the commas. The keys are already known, they come from the function
definition, so there is no reason to let the model write them.

The model only produces the values, and the allowed tokens depend on
the type of the parameter:

| Type | Allowed tokens |
|---|---|
| `string` | letters, digits, spaces and punctuation, except `"` |
| `number` | digits, `.` and `-` only |

For a string, the program writes the opening quote, generates the
value, then writes the closing quote itself. Generation stops when the
model chooses a closing token, or after the maximum number of tokens.

For a number, a token is kept only if the text stays a valid number.
The value is converted with `float()` at the end.

Because the structure is written by the program and the values are
constrained by type, the result is always valid JSON that matches the
function definition.

## Design decisions

- **The program writes the structure, the model writes the values.**
  This removes every syntax error at the source. The model cannot
  forget a quote or a comma, because it never writes them.
- **One token set per type.** The valid ids for strings and for
  numbers are computed from the vocabulary. This enforces the schema,
  not only the JSON syntax: a `number` parameter can never receive
  letters.
- **A few examples in the argument prompt.** `build_argument_prompt()`
  shows three requests with their arguments. This helps a 0.6B model
  copy the values from the request instead of inventing them.
- **A limit on every loop.** Each generation loop has a maximum number
  of steps, so an edge case can never freeze the program.
- **One entry per prompt, in order.** If a prompt fails, the error is
  printed and an empty entry is still written, so the results stay
  aligned with the input file.
- **pydantic models** in `src/models.py` validate the input files and
  the output entries.
- **The bonuses are separate.** The visualizer is its own module with
  a flag, so the normal code path stays readable.

## Performance analysis

- **Accuracy:** 11/11 on the public set of the moulinette.
- **Validity:** every output is valid JSON with the expected keys and
  types. This is guaranteed by construction, not by chance, so a
  parsing error is not possible.
- **Reliability:** the function name is always one of the defined
  names. Simple values, such as a name or a short string, are
  extracted reliably. The regular expression cases were the last ones
  to work, because the model has to produce a pattern instead of
  copying text from the request.
- **Speed:** loading the model takes most of the time. After that,
  each token needs one forward pass, and each value needs a few
  tokens, so a prompt is processed in a few seconds on CPU.

## Challenges faced

- **Spaces written as `Ġ`.** The tokenizer stores a space as the
  character `Ġ` in the vocabulary. While that character was not
  allowed, the values came out without their spaces. The fix was to
  allow it during generation and to convert it back to a real space at
  the end.
- **String values that did not stop.** The tokenizer merges the
  closing quote with the following character into single tokens such
  as `",` and `"}`. Looking for a lone `"` token was therefore not
  enough, and the value kept going into the rest of the JSON.
  `get_closing_tokens()` now treats a token as the end of the string
  when it starts with `"` and has no letter or digit after it. That
  covers `"`, `",` and `"}`, but not a token such as `"name` which
  begins the next key.
- **Disk quota.** The 42 machines have a small quota. Installing torch
  and downloading the model barely fits, so the caches had to be
  cleaned regularly.
- **torch on macOS Intel.** Recent versions of torch do not support
  that platform, which made the local setup harder.

## Testing strategy

The program was run on the provided prompts, and the output file was
graded with the moulinette:

```bash
cd moulinette
uv sync
uv run python -m moulinette prepare_exercises --set public

cd ..
uv run python -m src \
    --functions_definition moulinette/data/input/functions_definition.json \
    --input moulinette/data/input/function_calling_tests.json

cd moulinette
uv run python -m moulinette grade_student_answers --set public \
    --student_answer_path ../data/output/function_calling_results.json
```

The error handling was checked by hand: a missing input file, a file
that contains invalid JSON, a file whose structure does not match the
models, and an interruption with `Ctrl+C`. In each case the program
prints a clear message instead of a traceback.

The `--visualize` option was also used as a debugging tool. It prints,
for each step of the function name generation, how many tokens were
allowed and which one was chosen. That made it easy to see where a
wrong answer came from.

## Resources

- [Summary of the tokenizers][tokenizers] (Hugging Face)
- [Qwen3-0.6B model card][qwen]
- [Let's build the GPT Tokenizer][karpathy] (Andrej Karpathy)
- [Large Language Models explained][3b1b] (3Blue1Brown)
- [JSON specification][json]
- [pydantic documentation][pydantic]
- [uv documentation][uv]

[tokenizers]: https://huggingface.co/docs/transformers/tokenizer_summary
[qwen]: https://huggingface.co/Qwen/Qwen3-0.6B
[karpathy]: https://www.youtube.com/watch?v=zduSFxRajkE
[3b1b]: https://www.youtube.com/watch?v=LPZh9BOjkQs
[json]: https://www.json.org/
[pydantic]: https://docs.pydantic.dev/
[uv]: https://docs.astral.sh/uv/

### How AI was used

An AI assistant (Claude) was used as a tutor and as a review tool, not
as a way to skip the work. It was used for:

- **Understanding the theory.** Explanations of tokenization, logits
  and constrained decoding, with small worked examples.
- **Debugging.** Finding why the values lost their spaces (the `Ġ`
  character) and why the string values did not stop (the merged
  closing quote tokens).
- **Environment problems.** Working around the disk quota on the 42
  machines and the torch versions that do not support macOS Intel.
- **Code review.** Reading `src/decoder.py` and
  `src/function_selector.py` to point out unclear names and missing
  error cases.
- **Writing.** Drafting the docstrings and this README.

Every explanation was checked against the real code and the real
output before being kept, and the final result was verified with the
moulinette.