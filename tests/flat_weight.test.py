import random
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web3 import Web3, HTTPProvider
from client import Client

if __name__ == '__main__':
    web3 = Web3(HTTPProvider('http://localhost:8545'))
    c = Client(0, None, None, web3)
    c.setup_model('perceptron')
    weights_metadata = c.weights_metadata

    weights = {
        name: np.random.random(shape) for name, (shape, _) in weights_metadata.items()
    }
    total_size = sum(v.size for _, v in weights.items())

    flattened = c.flatten_weights(weights)
    assert len(flattened) == total_size

    unflattened = c.unflatten_weights(flattened)
    for key1, key2 in zip(sorted(unflattened), sorted(weights)):
        assert key1 == key2
        assert np.array_equiv(unflattened[key1], weights[key2])

    print("Tests passed!")
