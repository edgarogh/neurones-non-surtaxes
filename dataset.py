from typing import List, Tuple

from tokenizer import Tokenizer, TokenSet

WORD_BLACKLIST = [
    "les",
    "le",
    "des",
    "de"
]

in_file = open("dataset_test.csv", "r")

pairs: List[Tuple[str, bool]] = []

tokenizer = Tokenizer()

i = 0
for line in in_file:
    line = line.strip()
    if i != 0:
        given, expected = line.split(',')
        pairs.append((given, expected == 'breton'))
        tokenizer.get_tokens(given)
    i += 1

tokenizer.lock()
