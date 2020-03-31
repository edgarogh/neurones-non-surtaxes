import pickle
import sys

from neural_network import create_nn
from tokenizer import Tokenizer

data = pickle.load(open('model.bin', 'rb'))
layout = data["layout"]
weights = data["weights"]

tokenizer = Tokenizer(data["dictionary"])

sentence = ' '.join(sys.argv[1:])

call_nn = create_nn(layout, weights)


def wrap_nn(call_nn, tokenizer: Tokenizer):
    def call(sentence: str):
        input_layer = tokenizer.input_layer_of_sentence(sentence)
        output_layer = call_nn(input_layer)
        return output_layer[0]

    return call


sentence_nn = wrap_nn(call_nn, tokenizer)

output = sentence_nn(sentence)

bzh = output > 0

print(sentence + ': ' + ('Breton' if bzh else 'Français'))
