__credits__ = ["Milo Spencer-Harper", "Siraj Raval"]

from numpy import exp, array, random, dot

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

class NeuralNetwork():
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in xrange(number_of_training_iterations):
            # Pass the training set through our neural network.
            output_from_layer_1, output_from_layer_2, output_from_layer_3 = self.think(training_set_inputs)

            # Calculates the error from layer 3 since this is the last layer
            # (The difference between the desired output and the predicted output).
            layer3_error = training_set_outputs - output_from_layer_3
            layer3_delta = layer3_error * self.__sigmoid_derivative(output_from_layer_3)

            # Calculate the error for layer 2 (By looking at the weights in layer 2,
            # we can determine by how much layer 2 contributed to the error in layer 3).
            layer2_error = layer3_delta.dot(self.layer3.synaptic_weights.T)
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)
            
            # Calculate the error for layer 1
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            layer3_adjustment = output_from_layer_2.T.dot(layer3_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment
            self.layer3.synaptic_weights += layer3_adjustment

    # The neural network thinks.
    def think(self, inputs):
        # Pass inputs through our neural network.
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        output_from_layer3 = self.__sigmoid(dot(output_from_layer2, self.layer3.synaptic_weights))
        return output_from_layer1, output_from_layer2, output_from_layer3


if __name__ == "__main__":
    # Seed the random number generator, so it generates the same numbers
    # every time the program runs.
    random.seed(1)
    
    #Intialise a 3 layer neural network.
    #Layer1 = 4 neurons, each with 3 inputs
    #Layer2 = 2 neurons, each with 4 inputs
    #Layer3 = 1 neuron, with 2 inputs
    layer1 = NeuronLayer(4, 3)
    layer2 = NeuronLayer(2, 4)
    layer3 = NeuronLayer(1, 2)
    neural_network = NeuralNetwork(layer1, layer2, layer3)

    print "Random starting synaptic weights: "
    print "Layer 1 (4 neurons, each with 3 inputs): "
    print layer1.synaptic_weights
    print "Layer 2 (2 neuron, with 4 inputs):"
    print layer2.synaptic_weights
    print "Layer 3 (1 neuron, with 2 inputs):"
    print layer3.synaptic_weights
    print " "

    # The training set. We have 4 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print "New synaptic weights after training: "
    print "Layer 1 (4 neurons, each with 3 inputs): "
    print layer1.synaptic_weights
    print "Layer 2 (2 neuron, with 4 inputs):"
    print layer2.synaptic_weights
    print "Layer 3 (1 neuron, with 2 inputs):"
    print layer3.synaptic_weights
    print " "

    # Test the neural network with a new situation.
    print "Considering new situation [1, 1, 0] -> ?: "
    hidden_state1, hidden_state2, output = neural_network.think(array([1, 1, 0]))
    print output
