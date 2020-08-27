import numpy as np
import matplotlib.pyplot as plt
from utils import get_state_vowel

class HopfieldNetwork:
    """
    Creates a Hopfield Network.
    """
    def __init__(self, patterns):
        """
        Initializes the network.
        
        Args:
            patterns (np.array): Group of states to be memorized by the network.

        """
        self.num_units = patterns.shape[1]
        self.passes = 0
        self.state_units = np.array([1 if 2 * np.random.random() - 1 >= 0 else 0 for _ in range(self.num_units)])
        self.W = np.zeros((self.num_units, self.num_units))
        for pattern in patterns:
            self.W += np.dot(np.transpose((2 * patterns - 1)), (2 * patterns - 1))
        np.fill_diagonal(self.W, 0)
        self.energy = [-0.5 * np.dot(np.dot(self.state_units.T, self.W), self.state_units)]

    def _generate_sequence_units(self):
        """ Selects randomly the order to update states in the next iteration."""
        return np.random.choice(self.num_units, self.num_units)

    def run(self):
        """ Runs the network until no updates occur. """
        no_update = True
        while True:
            for unit in self._generate_sequence_units():
                unit_activation = np.dot(self.W[unit, :], self.state_units)
                if unit_activation >= 0 and self.state_units[unit] == 0:
                    self.state_units[unit] = 1
                    no_update = False
                elif unit_activation < 0 and self.state_units[unit] == 1:
                    self.state_units[unit] = 0
                    no_update = False
                self.energy.append(-0.5 * np.dot(np.dot(self.state_units.T, self.W), self.state_units)) 
            self.passes += 1
            if no_update:
                break
            else:
                no_update = True


if __name__ == "__main__":
    np.random.seed(1234)
    patterns = np.array([get_state_vowel('A'),
                         get_state_vowel('E'),
                         get_state_vowel('I'),
                         get_state_vowel('O'),
                         get_state_vowel('U')])
    net = HopfieldNetwork(patterns)
    net.run()

    # Plot patterns and output
    plt.figure(figsize=(6, 3), tight_layout=True)
    plt.subplot(2, 3, 1)
    plt.imshow(np.reshape(patterns[0, :], (5, 5)), cmap="Greys_r")
    plt.title("A")
    plt.subplot(2, 3, 2)
    plt.imshow(np.reshape(patterns[1, :], (5, 5)), cmap="Greys_r")
    plt.title("E")
    plt.subplot(2, 3, 3)
    plt.imshow(np.reshape(patterns[2, :], (5, 5)), cmap="Greys_r")
    plt.title("I")
    plt.subplot(2, 3, 4)
    plt.imshow(np.reshape(patterns[3, :], (5, 5)), cmap="Greys_r")
    plt.title("O")
    plt.subplot(2, 3, 5)
    plt.imshow(np.reshape(patterns[4, :], (5, 5)), cmap="Greys_r")
    plt.title("U")
    plt.subplot(2, 3, 6)
    plt.imshow(np.reshape(net.state_units, (5, 5)), cmap="Greys_r")
    plt.title("Output")

    # Plot energy over time
    plt.figure(figsize=(4, 2))
    plt.plot(net.energy)
    plt.title("Energy")
    plt.show()
