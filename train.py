import pickle
import select
import sys
from random import random
from typing import List, Tuple

from dataset import TokenSet, dictionary, get_tokens, pairs
from neural_network import create_nn, weights_length
from tailrec import recurse, tail_recursive

# [Input layer, Intermediary layer 1, ..., Output layer]
nn_layout = [len(dictionary), 6, 8, len(dictionary)]
nn_layout = [len(dictionary), 6, 8, len(dictionary)]


def wrap_nn(call_nn, dictionary: list):
    def call(tokens: list):
        input_layer = [1.0 if index in tokens else -1.0 for index in range(len(dictionary))]
        output_layer = call_nn(input_layer)
        returned_tokens = [(dictionary[i], weight) for (i, weight) in enumerate(output_layer)]
        return returned_tokens

    return call


def more_less(value: float, amount: float) -> float:
    delta = (random() * 2 - 1) * amount
    value += delta
    return max(-1, min(1, value))


def mutate(weights: List[float], amount: float) -> List[float]:
    return [more_less(weight, amount) for weight in weights]


weights = [0] * weights_length(nn_layout)
call_nn = create_nn(nn_layout, weights)


print("Dataset size: {}; Dictionary size: {}; Nodes: {}; Connections: {}".format(len(pairs), len(dictionary), sum(nn_layout), weights_length(nn_layout)))


def transform_weight(x: float) -> float:
    return x ** 2.0


bzh = dictionary.index('breton')
fr = dictionary.index('français')


def calc_fitness(call_nn):
    import sys
    total = 0.0

    for (given, expected) in pairs:
        input_layer = [1.0 if index in given else -1.0 for index in range(len(dictionary))]
        output_layer = call_nn(input_layer)
        tokens = [(dictionary[i], weight) for (i, weight) in enumerate(output_layer)]

        bExpected = expected[0] == bzh

        _, b = tokens[bzh]
        _, f = tokens[fr]

        b += 1
        f += 1

        b /= 2
        f /= 2

        delta = abs(b - f)

        bFound = b > f

        if bFound == bExpected:
            total += delta
        else:
            total += -delta

    return total


@tail_recursive
def train(siblings: int, generations: int, create_nn, weights: List[float], best_fitness: float) -> Tuple[float, List[float]]:
    print("Gen " + str(generations) + ', Accuracy: ' + str(100 * best_fitness / len(pairs)) + '%')
    best_weights = weights
    best_fitness = best_fitness # Useless but looks better

    for _ in range(siblings):
        weights_new = mutate(weights, 0.1)
        nn = create_nn(nn_layout, weights_new)
        fitness = calc_fitness(nn)
        if fitness > best_fitness:
            best_weights = weights_new
            best_fitness = fitness

    if generations == 0 or select.select([sys.stdin], [], [], 0.0)[0]:
        return (best_fitness, best_weights)
    else:
        return recurse(siblings, generations - 1, create_nn, best_weights, best_fitness)


best_fitness, best_weights = train(10, 500, create_nn, weights, float('-inf'))
print("Best fitness: " + str(best_fitness))

data = {
    "dictionary": dictionary,
    "layout": nn_layout,
    "weights": best_weights,
}

pickle.dump(data, open('model.bin', 'wb'))
