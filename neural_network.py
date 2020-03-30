from typing import Callable, List

Layout = List[int]


def memoize(f):
    memo = {}

    def helper(x, y):
        if (x, y) not in memo:
            memo[(x, y)] = f(x, y)
        return memo[(x, y)]

    return helper


def sigmoid(x: float) -> float:
    import math
    return 2.0 / (1.0 + math.exp(-x)) - 1.0


def weights_length(layout: Layout):
    sum = 0

    for (h2, h1) in zip(layout[1:], layout):
        sum += h1 * h2

    return sum


def create_nn(layout: Layout, weights: List[float]) -> Callable[[List[float]], List[float]]:
    layout_weights = []
    i = 0

    # Iterate on consecutive values of "layout"
    for (h2, h1) in zip(layout[1:], layout):
        span = h1 * h2
        layout_weights.append(weights[i:i+span])
        i += span

    assert i == weights_length(layout)

    def get_weight(layer_index: int, index_1: int, index_2: int):
        layer_weights = layout_weights[layer_index]
        return layer_weights[layout[layer_index] * index_2 + index_1]

    def call_nn(input_layer: List[float]) -> List[float]:
        assert(len(input_layer) == layout[0])

        @memoize
        def node_value(layer: int, index: int) -> float:
            if layer == 0:
                return input_layer[index]
            else:
                prev_layer = layer - 1
                prev_layer_size = layout[prev_layer]

                sum = 0
                for i in range(prev_layer_size):
                    sum += node_value(prev_layer, i) * \
                        get_weight(prev_layer, i, index)

                return sigmoid(sum)

        last_layer = len(layout) - 1
        #returned_tokens = [(dictionary[i], node_value(last_layer, i))
        #                   for i in range(layout[last_layer])]
        return [node_value(last_layer, i) for i in range(layout[last_layer])]

    return call_nn
