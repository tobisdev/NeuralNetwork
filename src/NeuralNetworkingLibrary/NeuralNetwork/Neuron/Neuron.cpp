//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 14.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#include "../NeuralNetwork.h"

Neuron::Neuron(int weightCount) {
    for (int i = 0; i < weightCount; ++i) {
        this->weights.push_back(0.0);
    }
    this->bias = 1.0;
}

Neuron::Neuron(std::vector<double> &weightList) {
    for (int i = 0; i < weightList.size(); ++i) {
        this->weights.push_back(weightList.at(i));
    }
}



void Neuron::reset() {
    this->bias = 1.0;
    this->nodeIn = 0.0;
    this->nodeOut = 0.0;
    this->delta = 0.0;
    for (int i = 0; i < weights.size(); ++i) {
        weights.at(i) = 0.0;
    }
}

void Neuron::randomizeWeights(double min, double max) {
    for (int i = 0; i < this->weights.size(); ++i) {
        double f = (double)rand() / RAND_MAX;
        this->weights.at(i) = min + f * (max - min);
    }
}

void Neuron::mutateWeights(double min, double max) {
    for (int i = 0; i < this->weights.size(); ++i) {
        double f = (double)rand() / RAND_MAX;
        this->weights.at(i) *= min + f * (max - min);
    }
}

void Neuron::setWeight(int position, double weightValue){
    this->weights.at(position) = weightValue;
}

double Neuron::getWeight(int positions) {
    return this->weights.at(positions);
}

void Neuron::setBias(double biasValue) {
    this->bias = biasValue;
}

double Neuron::getBias() {
    return this->bias;
}

void Neuron::setNodeIn(double inValue) {
    this->nodeIn = inValue;
}

double Neuron::getNodeIn() {
    return this->nodeIn;
}

void Neuron::setNodeOut(double inValue) {
    this->nodeOut = inValue;
}

double Neuron::getNodeOut() {
    return this->nodeOut;
}

int Neuron::getSize(){
    return weights.size();
}

double Neuron::getDelta() {
    return this->delta;
}

void Neuron::setDelta(double deltaValue) {
    this->delta = deltaValue;
}

void Neuron::print() {
    std::cout << "nodeIn: " << nodeIn << "; nodeOut: " << nodeOut << "; Bias: " << bias << "; Error: " << delta << ";" << ";\n";
    std::cout << "[:--:] ";
    for (int i = 0; i < weights.size(); ++i) {
        std::cout << "Weight" << i << ": " << weights.at(i) << "; ";
    }
    std::cout << "\n";
}