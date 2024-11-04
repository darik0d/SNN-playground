##########################################
#   This file contains the scratch code  #
#   for the Spiking Neural Network       #
#   (SNN) implementation.                #
##########################################
import random


class LIFNeuron:
    def __init__(self, threshold=1.0, decay=0.95, reset_value=0.0):
        """
        Initializes a Leaky Integrate-and-Fire (LIF) neuron.
        :param threshold: Firing threshold.
        :param decay: Decay factor for potential.
        :param reset_value: Reset value after firing.
        """
        self.threshold = threshold
        self.decay = decay
        self.reset_value = reset_value
        self.potential = 0.0
        self.weights = []  # Each neuron will have weights based on input size

    def receive_input(self, input_data):
        """
        Updates the neuron's potential based on weighted input.
        :param input_data: List of binary inputs (1 or 0).
        :return: Boolean indicating if neuron fired.
        """
        # Calculate input as a dot product with weights
        weighted_input = sum(w * i for w, i in zip(self.weights, input_data))

        # Update potential with decay and weighted input
        self.potential = self.potential * self.decay + weighted_input

        # Check if neuron fires
        if self.potential >= self.threshold:
            self.potential = self.reset_value  # Reset potential after firing
            return True  # Neuron fires
        return False  # No spike

    def adjust_weights(self, input_data, label, learning_rate=0.1):
        """
        Adjusts the neuron's weights based on target label.
        :param input_data: Binary input data.
        :param label: Target label (0 or 1).
        :param learning_rate: Rate at which weights are adjusted.
        """
        for i in range(len(self.weights)):
            # Adjust weights based on error between current firing and label
            self.weights[i] += learning_rate * (label - self.potential) * input_data[i]


class SpikingNeuralNetwork:
    def __init__(self, input_size, num_classes):
        """
        Initializes the SNN with neurons per class.
        :param input_size: Size of input pattern.
        :param num_classes: Number of classification classes.
        """
        self.neurons = [LIFNeuron() for _ in range(num_classes)]
        for neuron in self.neurons:
            neuron.weights = [random.uniform(-1, 1) for _ in range(input_size)]

    def classify(self, input_data):
        """
        Classifies input data by firing the most responsive neuron.
        :param input_data: Binary input pattern.
        :return: Predicted class based on firing.
        """
        spikes = [neuron.receive_input(input_data) for neuron in self.neurons]
        return spikes.index(True) if any(spikes) else -1  # Return index of spiked neuron or -1

    def train(self, data, labels, learning_rate=0.1, epochs=10):
        """
        Trains the SNN with given data and labels.
        :param data: List of binary input patterns.
        :param labels: Corresponding class labels.
        :param learning_rate: Learning rate for weight adjustment.
        :param epochs: Number of training epochs.
        """
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for input_data, label in zip(data, labels):
                prediction = self.classify(input_data)
                if prediction != label:
                    self.neurons[label].adjust_weights(input_data, 1, learning_rate)
                    if prediction >= 0:
                        self.neurons[prediction].adjust_weights(input_data, 0, learning_rate)


# Generate synthetic binary dataset
def generate_data(num_samples=100, input_size=8):
    """
    Generates a dataset of binary patterns for classification.
    Class 0: Patterns with an even number of ones.
    Class 1: Patterns with an odd number of ones.

    :param num_samples: Number of samples to generate.
    :param input_size: Number of bits in each pattern.
    :return: List of binary patterns and corresponding labels.
    """
    data = []
    labels = []
    for _ in range(num_samples):
        # Generate a random binary pattern of specified length
        pattern = [random.randint(0, 1) for _ in range(input_size)]
        data.append(pattern)

        # Determine label based on the parity of the number of ones
        label = sum(pattern) % 2  # 0 for even, 1 for odd
        labels.append(label)

    return data, labels


# Example usage
if __name__ == "__main__":
    # Parameters
    input_size = 8  # Number of input bits per pattern
    num_classes = 2  # We have two classes: even (0) and odd (1)
    num_samples = 100  # Number of samples in the dataset
    epochs = 20  # Number of training epochs

    # Generate dataset
    data, labels = generate_data(num_samples=num_samples, input_size=input_size)

    # Initialize SNN
    snn = SpikingNeuralNetwork(input_size=input_size, num_classes=num_classes)

    # Train the SNN
    print("Training the SNN...")
    snn.train(data, labels, learning_rate=0.1, epochs=epochs)

    # Test the SNN
    print("\nTesting the SNN...")
    correct = 0
    for input_data, label in zip(data, labels):
        prediction = snn.classify(input_data)
        if prediction == label:
            correct += 1
        print(f"Input: {input_data} | Predicted: {prediction} | Actual: {label}")

    # Calculate accuracy
    accuracy = correct / len(data) * 100
    print(f"\nAccuracy: {accuracy:.2f}%")

