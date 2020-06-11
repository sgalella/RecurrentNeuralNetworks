import numpy as np


def create_xor_data(length):
    # Create inputs
    xor = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 0]])
    xor_inputs = np.zeros((length, 12), dtype=np.int)
    for row in range(length):
        xor_inputs[row, :] = np.random.permutation(xor).flatten()
    xor_inputs = xor_inputs.flatten()

    # Create outputs
    xor_outputs = np.zeros((xor_inputs.shape), dtype=np.int)
    xor_outputs[0:-1] = xor_inputs[1:]
    xor_outputs[-1] = xor_inputs[0]
    return xor_inputs, xor_outputs
