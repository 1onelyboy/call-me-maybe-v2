from src.vocab import load_vocab

vocab = load_vocab("data/input/vocab.json")

function_names = ['fn_add_numbers', 'ft_greet']

generated = ""

valid_ids = set()

for name in function_names:
    if name.startswith(generated):
        next_part = name[len(generated):]
        for token_str, token_id in vocab.items():
            if next_part.startswith(token_str):
                valid_ids.add(token_id)

print(valid_ids)