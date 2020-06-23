import numpy as np

XOR_GATE = np.array([[1, 0, 1], [0, 0, 0], [0, 1, 1], [1, 1, 0]])

STATE_VOWELS = {
    'A': [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
    'E': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    'I': [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    'O': [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    'U': [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
}


def create_xor_data(length):
    # Create inputs
    xor_inputs = np.zeros((length, 12), dtype=np.int)
    for row in range(length):
        xor_inputs[row, :] = np.random.permutation(XOR_GATE).flatten()
    xor_inputs = xor_inputs.flatten()

    # Create outputs
    xor_outputs = np.zeros((xor_inputs.shape), dtype=np.int)
    xor_outputs[0:-1] = xor_inputs[1:]
    xor_outputs[-1] = xor_inputs[0]
    return xor_inputs, xor_outputs


def get_state_vowel(letter):
    return STATE_VOWELS[letter]