import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
	def __init__(self, patterns, num_iterations):
		self.num_units = patterns.shape[1]
		self.num_iterations = num_iterations
		self.state_units = np.array([1 if 2 * np.random.random() - 1 >= 0 else 0 for _ in range(self.num_units)])
		self.W = np.zeros((self.num_units, self.num_units))
		for pattern in patterns:
			self.W += np.dot(np.transpose((2 * patterns - 1)), (2 * patterns - 1))
		np.fill_diagonal(self.W, 0)
		self.energy = [-0.5 * np.dot(np.dot(self.state_units.T, self.W), self.state_units)]

	def __generate_sequence_units(self):
		return np.random.choice(self.num_units, self.num_units)

	def run(self):
		no_update = True
		while True:
			for unit in self.__generate_sequence_units():
				unit_activation = np.dot(self.W[unit, :], self.state_units)
				if unit_activation >= 0 and self.state_units[unit] == 0:
					self.state_units[unit] = 1
					no_update = False
				elif unit_activation < 0 and self.state_units[unit] == 1:
					self.state_units[unit] = 0
					no_update = False
				self.energy.append(-0.5 * np.dot(np.dot(self.state_units.T, self.W), self.state_units))	
			if no_update:
				break
			else:
				no_update = True


if __name__ == "__main__":
	np.random.seed(1234)
	patterns = np.array([[1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
						 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
	net = HopfieldNetwork(patterns, 10)
	net.run()
	plt.figure(figsize=(6, 3))
	plt.subplot(1, 3, 1)
	plt.imshow(np.reshape(patterns[0, :], (4, 4)), cmap="Greys_r")
	plt.title("Pattern 1")
	plt.subplot(1, 3, 2)
	plt.imshow(np.reshape(patterns[1, :], (4, 4)), cmap="Greys_r")
	plt.title("Pattern 2")
	plt.subplot(1, 3, 3)
	plt.imshow(np.reshape(net.state_units, (4, 4)), cmap="Greys_r")
	plt.title("Output")
	plt.figure(figsize=(3, 2))
	plt.plot(net.energy)
	plt.title("Energy")
	plt.show()
