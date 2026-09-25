*This project has been created as part of the 42 curriculum by trohain.*

# call me maybe

## Description

A function calling tool: it translates natural language requests into
structured function calls, using the small `Qwen/Qwen3-0.6B` model.

```
"What is the sum of 2 and 3?"  ->  {"name": "fn_add_numbers",
                                    "parameters": {"a": 2.0, "b": 3.0}}
```

The program does not answer the question. It picks the right function
and extracts its arguments with the correct types. Small models are
unreliable at producing JSON on their own, so every token is generated
with **constrained decoding**: invalid tokens are removed before the
model is allowed to choose, which guarantees valid, schema-compliant
output.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                  # install dependencies
uv run python -m src     # run with the default paths
```

The first run downloads the model (about 1.5 GB) into the Hugging Face
cache. On Linux, `pyproject.toml` installs the CPU-only build of torch,
which avoids several GB of CUDA packages.

### Makefile rules

| Rule | What it does |
|---|---|
| `make install` | install the dependencies (`uv sync`) |
| `make run` | run the program |
| `make debug` | run the program under `pdb` |
| `make clean` | remove `__pycache__` and `.mypy_cache` |
| `make lint` | `flake8` + `mypy` with the flags required by the subject |
| `make lint-strict` | `flake8` + `mypy --strict` |

## Example usage

```bash
# default paths
uv run python -m src

# custom paths
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json

# show every decoding step (bonus)
uv run python -m src --visualize

# list all options
uv run python -m src --help
```

`--model` selects another Hugging Face model (default `Qwen/Qwen3-0.6B`).
Only Qwen3-0.6B has been tested.

## Project structure

```
.
├── src/
│   ├── __main__.py            entry point: options, pipeline, output file
│   ├── models.py              pydantic models for inputs and output
│   ├── loader.py              loads and validates the input JSON files
│   ├── vocab.py               loads the tokenizer vocabulary
│   ├── function_selector.py   picks the function (constrained decoding)
│   ├── decoder.py             generates typed arguments (constrained decoding)
│   └── visualizer.py          step-by-step view of the decoding (bonus)
├── llm_sdk/                   provided SDK (Small_LLM_Model)
├── data/input/                input files
├── Makefile
├── pyproject.toml
└── uv.lock
```

How the modules work together for one prompt:

```
loader.py ──> prompts + function definitions (validated by models.py)
                 │
                 ▼
function_selector.py ──> "fn_add_numbers"
                 │
                 ▼
decoder.py ──> {"a": 2.0, "b": 3.0}
                 │
                 ▼
__main__.py ──> one entry in data/output/function_calling_results.json
```

## Input and output files

**`functions_definition.json`** lists the functions that can be called:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

**`function_calling_tests.json`** lists the requests to process:

```json
[{"prompt": "What is the sum of 2 and 3?"}]
```

**`function_calling_results.json`** (output) contains exactly one entry
per prompt, in the same order, with exactly three keys:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

Supported parameter types: `number` (written as a float), `integer`,
`string` and `boolean`.

## Algorithm explanation

Generation is done token by token. At each step the model returns a
score (logit) for every token of its ~150k-token vocabulary. Before
picking, we keep only the tokens that are valid at this point, and take
the best of those. The vocabulary file (`get_path_to_vocab_file()`)
maps each token to its text, which is how we decide which tokens are
valid.

**1. Function selection** (`src/function_selector.py`)

The prompt lists the available functions. We then generate the name
token by token. A token is allowed only if the text generated so far
plus that token is still the beginning of a real function name.

```
generated ""              -> allowed: "fn_"
generated "fn_"           -> allowed: "add", "greet", "reverse", "get" ...
generated "fn_greet"      -> complete name, stop
```

The output can therefore only ever be one of the defined function names.
The model's understanding of the request decides between the options.

**2. Argument generation** (`src/decoder.py`)

The JSON structure is written by the program, never by the model:
`{`, the keys, the quotes, `:` and `,`. The model only fills in values,
and the allowed tokens depend on the parameter type:

| Type | Allowed tokens | Stops when |
|---|---|---|
| `number` | digits, `.`, `-`, only if the text stays a valid number | the model picks `,` or `}` |
| `string` | printable characters except `"` | the model picks the closing `"` |
| `boolean` | `true` or `false` | after one choice |

Because the structure is fixed and the values are type-constrained, the
result is always parseable and matches `functions_definition.json`.

## Design decisions

- **Structure written by code, values by the model.** Keys are already
  known from the function definition, so asking the model for them only
  adds ways to fail.
- **Type-based token sets.** This enforces the schema, not only JSON
  syntax: a `number` field can never receive text.
- **Few-shot argument prompt.** Showing a few example requests with
  their arguments strongly improves extraction for a 0.6B model.
- **Safety caps.** Every generation loop has a maximum number of steps,
  so a tokenizer edge case can never freeze the program.
- **One output entry per prompt, in order.** If a prompt fails, an
  entry is still written, so the results stay aligned with the input.
- **pydantic models** validate the input files and the output entries.

## Error handling

The program never crashes with a traceback. It prints a clear message
and exits with status 1 for:

- a missing input file, a path that is a directory, no read permission
- invalid JSON, or valid JSON with the wrong structure (checked by pydantic)
- a model or vocabulary that cannot be loaded
- an output file that cannot be written
- `Ctrl+C`

If a single prompt fails, the error is printed and the remaining
prompts are still processed.

## Bonus features

**Visualization of the generation process** (`--visualize`). For every
generated token it shows how many tokens were allowed, the three best
allowed candidates with their scores, the chosen token, and the text so
far. Spaces are shown as `·`. Example:

```
[1/11] What is the sum of 2 and 3?
         function      |      2 allowed | best: 'fn_' 21.4, 'f' 9.8
                       | chose 'fn_' -> so far: 'fn_'
         function      |      9 allowed | best: 'add' 18.2, 'get' 11.7, 'greet' 6.3
                       | chose 'add' -> so far: 'fn_add'
         ...
         a             |     12 allowed | best: '2' 16.9, '3' 12.1, '1' 8.0
                       | chose '2' -> so far: '2'
         a             | stop: end of number
```

**Caching.** Filtering the 150k-token vocabulary into the string and
number token sets is done once and reused for every prompt.

## Performance analysis

- **Validity:** 100% of outputs are valid JSON with the right keys and
  types, guaranteed by construction.
- **Function selection:** 11/11 correct on the provided tests.
- **Arguments:** simple extractions (names, strings, numbers) are
  reliable. The regex tests are the hardest, because the model must
  invent a pattern instead of copying text from the request.
- **Speed:** model loading dominates; processing the provided prompts
  takes well under the 5 minute limit on CPU.

*(Update with your final moulinette score and timing.)*

## Challenges faced

- **Tokenizer markers.** Spaces are stored as `Ġ` and newlines as `Ċ` in
  the vocabulary. Blocking `Ġ` made the model drop spaces or replace
  them with `.` and `,`; allowing it and converting it back fixed that.
- **Values leaking into the JSON.** Before the character filter, string
  values could swallow the closing quote and the next keys.
- **Disk space.** The school machines have a 4.7 GB quota; the model,
  torch and the uv cache barely fit, so caches had to be cleaned.
- **torch versions.** Recent torch does not support macOS Intel; on
  Linux a CPU-only index avoids downloading several GB of CUDA packages.

## Testing strategy

- Ran the program on the provided prompts and compared every result to
  the values expected by the moulinette.
- Tested the decoding logic offline with a fake model that scores
  tokens, to check stopping on quotes, number termination, space
  handling and output writing without loading the real model.
- Tested error handling: missing files, invalid JSON, wrong structure,
  and interruption with `Ctrl+C`.
- Graded the output with the moulinette:

```bash
cd moulinette
uv sync
uv run python -m moulinette grade_student_answers --set private \
    --student_answer_path ../data/output/function_calling_results.json
```

## Resources

- [Hugging Face - Tokenizers summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [Andrej Karpathy - Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
- [3Blue1Brown - Large Language Models explained](https://www.youtube.com/watch?v=LPZh9BOjkQs)
- [JSON specification](https://www.json.org/)
- [pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)

### How AI was used

Claude (Anthropic) was used as a tutor throughout the project: to
explain tokenization, logits and constrained decoding with worked
examples, to debug environment issues (torch versions, disk space), to
review code, and to draft parts of `decoder.py`, the visualizer and this
README. Every part was traced step by step with real data until
understood, and the results were checked by running the program and the
moulinette.