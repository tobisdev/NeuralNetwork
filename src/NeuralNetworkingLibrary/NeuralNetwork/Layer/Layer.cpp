//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 14.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#include "../NeuralNetwork.h"

Layer::Layer(int neuronCount, int weightCount, layerType type, activationFunctions activationFunction) {
    this->type = type;
    for (int i = 0; i < neuronCount; ++i) {
        this->neurons.push_back(Neuron(weightCount));
    }
}

void Layer::reset() {
    for (int i = 0; i < this->neurons.size(); ++i) {
        this->neurons.at(i).reset();
    }
}

void Layer::randomizeWeights(double min, double max) {
    for (int i = 0; i < this->neurons.size(); ++i) {
        this->neurons.at(i).randomizeWeights(min, max);
    }
}

void Layer::mutateWeights(double min, double max) {
    for (int i = 0; i < this->neurons.size(); ++i) {
        this->neurons.at(i).mutateWeights(min, max);
    }
}

void Layer::setType(layerType neuronType) {
    this->type = neuronType;
}

Neuron &Layer::getNeuron(int n) {
    return this->neurons.at(n);
}


layerType Layer::getType() {
    return this->type;
}

int Layer::getSize() {
    return this->neurons.size();
}

void Layer::setActivation(activationFunctions activationFunction) {
    this->activation = activationFunction;
}

activationFunctions Layer::getActivation() {
    return this->activation;
}

void Layer::print() {
    std::cout << "Type: " << std::to_string(type) << ";\n";
    for (int i = 0; i < neurons.size(); ++i) {
        std::cout << "[::] " << "Neuron" << i << ": ";
        neurons.at(i).print();
    }
    std::cout << "\n";
}