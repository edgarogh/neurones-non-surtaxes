import pickle
import sys

from dataset import get_tokens
from neural_network import create_nn

sentence = ' '.join(sys.argv[1:])
tokens = get_tokens(sentence)

data = pickle.load(open('model.bin', 'rb'))
dictionary = data["dictionary"]
layout = data["layout"]
weights = data["weights"]

call_nn = create_nn(layout, weights)


def wrap_nn(call_nn, dictionary: list):
    def call(tokens: list):
        input_layer = [1.0 if index in tokens else -
                       1.0 for index in range(len(dictionary))]
        output_layer = call_nn(input_layer)
        returned_tokens = [(dictionary[i], weight)
                           for (i, weight) in enumerate(output_layer)]
        return returned_tokens

    return call


sentence_nn = wrap_nn(call_nn, dictionary)

output = sentence_nn(tokens)

bzh = dictionary.index('breton')
fr = dictionary.index('français')

_, b = output[bzh]
_, f = output[fr]

print(sentence + ': ' + ('Breton' if b > f else 'Français'))
