//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 14.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {

}

NeuralNetwork::NeuralNetwork(std::vector<int> &topology, activationFunctions acIn, activationFunctions acHid, activationFunctions acOut) {             // Creating the Neural Network
    if(topology.size() < 2){
        std::cout << "ERROR: Neural Network must at least contain an input and an output layer!" << "\n";
        return;
    }

    this->layers.push_back(Layer(topology.at(0), 0, INPUTLAYER, acIn));                                                                             // Creating the input layer

    for (int i = 1; i < topology.size() - 1; ++i) {                                                                                               // If there are more than 2 Layers, create the hidden layers
        this->layers.push_back(Layer(topology.at(i), topology.at(i - 1), HIDDENLAYER, acHid));
    }

    this->layers.push_back(Layer(topology.at(topology.size() - 1), topology.at(topology.size() - 2), OUTPUTLAYER, acOut));                       // Creating the output layer
}

void NeuralNetwork::feedForward() {                                                                                                               // Updating the Neurons values by feeding their information forward
    for (int i = 0; i < this->layers.size(); ++i) {

        Layer &currentLayer = this->layers.at(i);

        for (int j = 0; j < currentLayer.getSize(); ++j) {

            Neuron &currentNeuron = currentLayer.getNeuron(j);

            if (i > 0) {                                                                                                                          // Calculate node input for hidden and output neurons
                Layer &previousLayer = this->layers.at(i - 1);
                double sum = 0.0;                                                                                                                 // Sum value, which stores the neurons node input value

                for (int k = 0; k < previousLayer.getSize(); ++k) {
                    sum += previousLayer.getNeuron(k).getNodeOut() * currentNeuron.getWeight(k);                                                  // Sum = ∑ PreviousNeuronOutputNode(i) * Weight(i)
                }

                currentNeuron.setNodeIn(sum + currentNeuron.getBias());                                                                           // Sum = Sum + Bias value

            }

            currentNeuron.setNodeOut(calculateActivation(currentLayer.getActivation(), false,currentNeuron.getNodeIn()));
        }
    }
}

void NeuralNetwork::backPropagation(std::vector<double> &desiredValues) {                                                                              // Calculating error between the output layer and the desired values
    if(desiredValues.size() != this->layers.at(layers.size() -1).getSize()){
        std::cout << "ERROR: Desired values have to be the same size as the output neurons!" << "\n";
        return;
    }

    for (int i = this->layers.size() - 1; i > 0; --i) {                                                                                           // Reverting from the output layer into the network
        // The input layer will get excluded
        Layer &currentLayer = this->layers.at(i);

        for (int j = 0; j < currentLayer.getSize(); ++j) {

            Neuron &currentNeuron = currentLayer.getNeuron(j);
            double delta = 0.0;                                                                                                                   // This value is not the direct error of the neuron but it's δ
            // corresponding to the error made by the weights
            if(i == layers.size() - 1){
                delta = 2 * (desiredValues.at(j) - currentNeuron.getNodeOut());                                                                // Calculating the absolute error between output neurons
            }else{                                                                                                                                // and the desired values
                Layer &earlierLayer = this->layers.at(i + 1);

                for (int k = 0; k < earlierLayer.getSize(); ++k) {
                    delta += earlierLayer.getNeuron(k).getDelta() * earlierLayer.getNeuron(k).getWeight(j);                                       // Adding up the sum of the errors of the neurons in front of the
                }                                                                                                                                 // current one
            }

            delta *= calculateActivation(currentLayer.getActivation(), true, currentNeuron.getNodeOut());

            currentNeuron.setDelta(delta);                                                                                                        // Updating the error value of the current neuron
        }
    }
}

void NeuralNetwork::updateValues(double learnRate, double momentum) {
    if (learnRate < 0 || learnRate > 1) {
        std::cout << "WARNING: Learn rate should be between 1 and 0!" << "\n";
    }

    if (momentum < 0 || momentum >= 1) {
        std::cout << "WARNING: Momentum should be between 0 and 1!" << "\n";
    }

    std::vector<std::vector<double>> weightUpdates(layers.size() - 1);
    std::vector<std::vector<double>> biasUpdates(layers.size() - 1);

    for (int i = 0; i < layers.size() - 1; ++i) {
        weightUpdates[i].resize(layers[i + 1].getSize(), 0.0);
        biasUpdates[i].resize(layers[i + 1].getSize(), 0.0);
    }

    for (int i = 1; i < layers.size(); ++i) {
        Layer &currentLayer = this->layers.at(i);
        Layer &previousLayer = this->layers.at(i - 1);

        for (int j = 0; j < currentLayer.getSize(); ++j) {
            Neuron &currentNeuron = currentLayer.getNeuron(j);

            for (int k = 0; k < previousLayer.getSize(); ++k) {
                // Compute weight and bias updates with momentum
                weightUpdates[i - 1][j] = momentum * weightUpdates[i - 1][j] +
                                          learnRate * currentNeuron.getDelta() * previousLayer.getNeuron(k).getNodeOut();
                biasUpdates[i - 1][j] = momentum * biasUpdates[i - 1][j] + learnRate * currentNeuron.getDelta();

                // Update weights and biases
                double newWeight = currentNeuron.getWeight(k) + weightUpdates[i - 1][j];
                double newBias = currentNeuron.getBias() + biasUpdates[i - 1][j];

                currentNeuron.setWeight(k, newWeight);
                currentNeuron.setBias(newBias);
            }
        }
    }
}

void NeuralNetwork::randomizeWeights(double min, double max) {
    for (int i = 0; i < this->layers.size(); ++i) {
        layers.at(i).randomizeWeights(min, max);
    }
}

void NeuralNetwork::mutateWeights(double min, double max) {
    for (int i = 0; i < this->layers.size(); ++i) {
        layers.at(i).mutateWeights(min, max);
    }
}

void NeuralNetwork::reset() {
    for (int i = 0; i < this->layers.size(); ++i) {
        this->layers.at(i).reset();
    }
}

void NeuralNetwork::setInput(std::vector<double> &input) {

    if(input.size() != layers.at(0).getSize()){
        std::cout << "ERROR: Input data must be the same size as the Number of input neurons!" << "\n";
        return;
    }

    for (int i = 0; i < layers.at(0).getSize(); ++i) {
        layers.at(0).getNeuron(i).setNodeIn(input.at(i));
    }
}

std::vector<double> NeuralNetwork::returnOutput() {
    std::vector<double> output;
    for (int i = 0; i < layers.at(layers.size() - 1).getSize(); ++i) {
        output.push_back(layers.at(layers.size() - 1).getNeuron(i).getNodeOut());
    }

    return output;
}

int NeuralNetwork::returnGuess() {
    int highestIndex = 0;
    double highestValue = 0.0;

    Layer &outputLayer = layers.at(layers.size() - 1);

    for (int i = 0; i < outputLayer.getSize(); ++i) {
        if(outputLayer.getNeuron(i).getNodeOut() > highestValue){
            highestValue = outputLayer.getNeuron(i).getNodeOut();
            highestIndex = i;
        }
    }

    return highestIndex;
}

Layer &NeuralNetwork::getLayer(int i) {
    return this->layers.at(i);
}

int NeuralNetwork::getSize() {
    return this->layers.size();
}

void NeuralNetwork::print() {
    std::cout << "NETWORK:\n" << "--------------------------------------------" << "\n";
    for (int i = 0; i < layers.size(); ++i) {
        std::cout << "[] " << "Layer" << i << ": ";
        layers.at(i).print();
    }
    std::cout << "\n";
}

void NeuralNetwork::updateTopology(std::vector<int> &topology) {
    if(topology.size() < 2){
        std::cout << "ERROR: Neural Network must at least contain an input and an output layer!" << "\n";
        return;
    }

    activationFunctions acIn = layers.at(0).getActivation();
    activationFunctions acHid = layers.at(1).getActivation();
    activationFunctions acOut = layers.at(layers.size() - 1).getActivation();

    layers.clear();

    this->layers.push_back(Layer(topology.at(0), 0, INPUTLAYER, acIn));                                                                             // Creating the input layer

    for (int i = 1; i < topology.size() - 1; ++i) {                                                                                               // If there are more than 2 Layers, create the hidden layers
        this->layers.push_back(Layer(topology.at(i), topology.at(i - 1), HIDDENLAYER, acHid));
    }

    this->layers.push_back(Layer(topology.at(topology.size() - 1), topology.at(topology.size() - 2), OUTPUTLAYER, acOut));
}

void NeuralNetwork::setInputFunction(activationFunctions ac) {
    this->layers.at(0).setActivation(ac);
}

void NeuralNetwork::setHidddenFunction(activationFunctions ac) {
    for (int i = 1; i < layers.size() - 1; ++i) {
        layers.at(i).setActivation(ac);
    }
}

void NeuralNetwork::setOutputFunction(activationFunctions ac) {
    layers.at(layers.size() - 1).setActivation(ac);
}

activationFunctions NeuralNetwork::getInputFunction() {
    return this->layers.at(0).getActivation();
}

activationFunctions NeuralNetwork::getHidddenFunction() {
    return this->layers.at(1).getActivation();
}

activationFunctions NeuralNetwork::getOutputFunction() {
    return this->layers.at(layers.size() - 1).getActivation();
}

double NeuralNetwork::calculateActivation(activationFunctions activationFunction, bool isDerivative, double value){
    if (isDerivative){
        switch(activationFunction){                                                                                                               // To calculate the error the derrivative for each of the activation
            case Sigmoid:                                                                                                                         // function is needed
                return value * (1 - value);                                                                                                       // This is the derrivative of the Sigmoid function
            case LeakyReLU:
                return (value > 0 ? 1 : 0.1);
            case ReLU:
                return (value > 0);
            case Binary:
                return (value > 0);
            case Linear:
                return 1;
            default:
                return 1;
        }
    }else{
        switch (activationFunction) {                                                                                                             // Activation Functions decide if a Neuron should fire or not
            case Sigmoid:                                                                                                                         // The Sigmoid function is the most common one
                return (1 / (1 + exp(-value)));                                                                                                // It is also called logistic function
            case LeakyReLU:
                return (value > 0 ? value : value * 0.1);
            case ReLU:                                                                                                                            // The ReLU function is a max(0, x) function
                return (value > 0 ? value : 0);
            case Binary:                                                                                                                          // The binary function either switches on or off
                return (value > 0 ? 1 : 0);
            case Linear:
                return value;
            default:
                return value;
        }
    }
}

double NeuralNetwork::calculateError(std::vector<double> & desiredValues) {
    if(desiredValues.size() != this->layers.at(layers.size() -1).getSize()){
        std::cout << "ERROR: Desired values have to be the same size as the output neurons!" << "\n";
        return -1.0;
    }

    double sum = 0.0;
    Layer &currentLayer = layers.at(layers.size() - 1);
    for (int i = 0; i < currentLayer.getSize(); ++i) {
        sum += pow(desiredValues.at(i) - currentLayer.getNeuron(i).getNodeOut(), 2);
    }
    return sum;
}

