from typing import List, Tuple

TokenSet = List[int]

WORD_BLACKLIST = [
    "les",
    "le",
    "des",
    "de"
]

in_file = open("dataset_test.csv", "r")

dictionary: List[str] = ["<RESERVED: PADDING>", "<RESERVED: UNKNOWN>"]
pairs: List[Tuple[TokenSet, TokenSet]] = []


def get_token(word: str, add_if_not_exist=False) -> int:
    try:
        return dictionary.index(word)
    except:
        if add_if_not_exist:
            dictionary.append(word)
            return len(dictionary) - 1
        else:
            return 1


def get_tokens(sentence: str, add_if_not_exist=False) -> TokenSet:
    words = [word.lower() for word in sentence.split()]
    return [get_token(word, add_if_not_exist) for word in words if (word not in WORD_BLACKLIST)]


i = 0
for line in in_file:
    if i != 0:
        l = line.split(',')
        given = l[0]
        expected = l[1]
        pairs.append((get_tokens(given, True), get_tokens(expected, True)))
    i += 1
