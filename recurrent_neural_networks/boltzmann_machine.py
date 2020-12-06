import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class BoltzmannMachine:
    """
    Creates a Boltzmann Machine.
    """
    def __init__(self, num_units, num_hidden):
        """
        Initializes the network.

        Args:
            num_units (int): Number of units.
            num_hidden (int): Number of hidden units.
        """
        self.num_units = num_units
        self.num_hidden = num_hidden
        self.idx_unclamped_plus = range(2 * self.num_units + 3, 2 * self.num_units + 3 + self.num_hidden)
        self.idx_unclamped_minus = range(0, 2 * self.num_units + 3 + self.num_hidden)
        self.W = 2 * np.random.random((2 * self.num_units + 3 + self.num_hidden + 1,
                                       2 * self.num_units + 3 + self.num_hidden + 1)) - 1
        self.W = (self.W + self.W.T) / 2 - np.diag(self.W.diagonal())  # Make W symmetric + no self-connections
        self.logicalW = self.logicalW = self.W.copy().astype(bool)

    def _generate_states(self, num_occurrences, initial_threshold):
        """
        Generates V1, V2 and V3 for the different number of occurences.

        Args:
            num_occurrences (int): Different number of shift vector.
            initial_threshold (float): Threshold of unit.

        Returns:
            np.array: Matrix with state vectors.
        """
        # Create matrices V1, V2 and V3 where each row of each is one occurence where
        # V3 specifies the shift of V1 represented in V2
        V1 = np.random.choice([0, 1], size=(num_occurrences, self.num_units), p=[0.7, 0.3])
        V3 = np.zeros((num_occurrences, 3))
        row = range(V3.shape[0])
        column = np.random.randint(3, size=(num_occurrences, ))
        V3[row, column] = 1
        V2 = np.array([np.roll(V1[i, :], column[i] - 1) for i in row])

        # Create matrix for hidden and threshold
        hidden = np.zeros((num_occurrences, self.num_hidden))
        threshold = initial_threshold * np.ones((num_occurrences, 1))

        # Concatenate all the states. Each matrix has num_ocurrences rows
        V = np.concatenate([V1, V2, V3, hidden, threshold], axis=1)
        V = np.expand_dims(V, axis=2)  # To slice column vectors of 2D
        return V

    def _update_state_unit(self, p, delta_E, k, V, temp):
        """
        Changes the state of a given unit.

        Args:
            p (float): Probability of spike.
            delta_E (float): Energy of the given unit.
            k (int): State unit.
            V (np.array): Vector of states.
            temp (float): Temperature of the system

        Returns:
            tuple: Updated state vector and energy of the unit.
        """
        new_Vk = p < 1 / (1 + np.exp(-delta_E[k] / temp))
        delta_Vk = new_Vk - V[k]
        if delta_Vk:
            delta_E += delta_Vk * self.W[:, k]
            V[k] = new_Vk
        return V, delta_E

    def _train(self, V, idx_unclamp, num_iterations, temp, temp_relax):
        """
        Trains a Boltzmann Network.

        Args:
            V (np.array): Vector of states.
            idx_unclamp (np.array): Indices of units to unclamp.
            num_iterations (int): Number of iterations in simulated annealing.
            temp (np.array): Temperature vector for simulated annealing.
            temp_relax (np.array): Temperature vector for relaxation phase.

        Returns:
            np.array: Phase state matrix.
        """
        P = np.zeros(self.W.shape)
        prob = iter(np.random.random((V.shape[0] * (num_iterations * len(idx_unclamp) * (len(temp) + len(temp_relax))))))
        for Vt in V:
            delta_E = np.matmul(self.W, Vt).flatten()
            for t in temp:
                for i in range(num_iterations):
                    random_idx = np.random.permutation(idx_unclamp)
                    for k in random_idx:
                        Vt, delta_E = self._update_state_unit(next(prob), delta_E, k, Vt, t)
            for t_r in temp_relax:
                for i in range(num_iterations):
                    random_idx = np.random.permutation(idx_unclamp)
                    for k in random_idx:
                        Vt, delta_E = self._update_state_unit(next(prob), delta_E, k, Vt, t_r)
                P += np.kron(Vt, Vt.T)
        P /= (V.shape[0] * len(temp_relax))
        return P

    def run(self, num_sweeps, num_iterations, num_occurrences, temp, temp_relax, initial_threshold,
            learning_rate, stretch_factor):
        """
        Runs the Boltzmann machine.

        Args:
            num_sweeps (int): Number of sweeps to run the algorithm.
            num_iterations (int): Number of iterations to repeat the simulated annealing.
            num_occurrences (int): Different number of shift vector.
            temp (np.array): Temperature vector for simulated annealing.
            temp_relax (np.array): Temperature vector for relaxation phase.
            initial_threshold (float): Threshold of unit.
            learning_rate (float): Update constant for the learning algorithm.
            stretch_factor (float): Squeze constant for the weight update.
        """
        # Main loop
        for _ in tqdm(range(num_sweeps)):
            V = self._generate_states(num_occurrences, initial_threshold)

            # Phase +
            P_plus = self._train(V, self.idx_unclamped_plus, num_iterations, temp, temp_relax)

            # Phase -
            P_minus = self._train(V, self.idx_unclamped_minus, num_iterations, temp, temp_relax)

            # Update W
            self.W += learning_rate * np.multiply(self.logicalW, (P_plus - P_minus))
            self.W *= (1 - stretch_factor)

    def plot_weights(self, rows, columns):
        """
        Plots the weight of the hidden units.

        Args:
            rows (int): Number of rows in the plot.
            columns (int): Number of columns in the plot.
        """
        fig, ax = plt.subplots(rows, columns, figsize=(15, 6))
        for i in range(rows * columns):
            current = self.W[2 * self.num_units + 3 + i][:]
            img = np.zeros((4, self.num_units))
            img[0][0] = current[-1]
            img[0][3:6] = current[2 * self.num_units:2 * self.num_units + 3]
            img[2] = current[:self.num_units]
            img[3] = current[self.num_units:2 * self.num_units]
            ax[i // columns, i % columns].imshow(img, cmap="Greys_r")
            ax[i // columns, i % columns].set_xticks([])
            ax[i // columns, i % columns].set_yticks([])
        plt.show()


def main():
    # Random seed (for reproducibility)
    np.random.seed(1234)

    # Initialize temperature for simulated annealing + relaxation
    relaxation_iterations = 10
    T = np.array([40, 40, 35, 35, 30, 30, 25, 25, 20, 20, 15, 15, 12, 12, 10, 10])
    T_relax = T[-1] * np.ones((relaxation_iterations, ))

    # Create BoltzmannMachine with num_units and num_hidden
    boltzmann = BoltzmannMachine(num_units=8, num_hidden=24)

    # Training
    boltzmann.run(num_sweeps=2000, num_iterations=2, num_occurrences=10, temp=T, temp_relax=T_relax,
                  initial_threshold=-1, learning_rate=5, stretch_factor=0.0005)

    # Plot the results
    boltzmann.plot_weights(rows=4, columns=6)


if __name__ == '__main__':
    main()
