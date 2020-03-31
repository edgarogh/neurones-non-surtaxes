import re
from typing import List, Tuple

TokenSet = List[int]

_WORD_SPLIT = re.compile(r'[ ,.!?;:\'"&%$€*/+-={}\[\]()]')


def _float_of_bool(b: bool) -> float:
    return 1.0 if b else -1.0


class Tokenizer:
    _dictionary: List[str] = ["<RESERVED: PADDING>", "<RESERVED: UNKNOWN>"]
    _blacklist: List[str] = []
    _locked = False

    def __init__(self, dictionary: List[str] = None, blacklist: List[str] = []):
        if dictionary is not None:
            self._dictionary = dictionary
            self.lock()
        self._blacklist = blacklist

    @property
    def dictionary_size(self) -> int:
        return len(self._dictionary)

    def export_dictionary(self):
        return self._dictionary

    def lock(self):
        self._locked = True

    def get_token(self, word: str) -> int:
        try:
            return self._dictionary.index(word)
        except:
            if not self._locked:
                self._dictionary.append(word)
                return len(self._dictionary) - 1
            else:
                return 1

    def get_tokens(self, sentence: str) -> TokenSet:
        words = _WORD_SPLIT.split(sentence)
        words = [word.lower() for word in words if word != ""]
        return [
            self.get_token(word)
            for word in words if word not in self._blacklist
        ]

    def input_layer_of_sentence(self, sentence: str) -> List[float]:
        tokens = self.get_tokens(sentence)
        return [
            _float_of_bool(index in tokens)
            for index in range(self.dictionary_size)
        ]

    def stats_of_output_layer(self, layer: List[float]) -> List[Tuple[str, float]]:
        return [
            (self._dictionary[i], weight)
            for (i, weight) in enumerate(layer)
        ]
