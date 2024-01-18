//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 14.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <iostream>
#include <thread>
#include <cmath>
#include <vector>

enum activationFunctions : int {Sigmoid = 0, LeakyReLU = 1, ReLU = 2, Binary  = 3, Linear = 4};
enum layerType : int {INPUTLAYER=0,HIDDENLAYER=1,OUTPUTLAYER=2};

class Neuron;
class Layer;

/*  This is the Neural Network it contains different functions and is built up of Neurons and Layers
 *
 */
class NeuralNetwork {
private:
    std::vector<Layer> layers;
    double calculateActivation(activationFunctions, bool, double);
public:
    NeuralNetwork();
    NeuralNetwork(std::vector<int> &, activationFunctions = Sigmoid, activationFunctions = Sigmoid, activationFunctions = Sigmoid);

    void updateTopology(std::vector<int> &);

    void setInputFunction(activationFunctions);
    void setHidddenFunction(activationFunctions);
    void setOutputFunction(activationFunctions);
    activationFunctions getInputFunction();
    activationFunctions getHidddenFunction();
    activationFunctions getOutputFunction();

    void feedForward();
    void backPropagation(std::vector<double> &desiredValues);
    void updateValues(double learnRate, double momentum);

    Layer &getLayer(int i);

    void reset();

    void print();

    void randomizeWeights(double, double);
    void mutateWeights(double, double);

    int returnGuess();
    std::vector<double> returnOutput();
    void setInput(std::vector<double> &);

    int getSize();

    double calculateError(std::vector<double> &desiredValues);
};


class Neuron {
private:
    std::vector<double> weights;
    double bias = 0.0;
    double nodeIn  = 0.0;
    double nodeOut = 0.0;
    double delta = 0.0;
public:
    Neuron();
    Neuron(int);
    Neuron(std::vector<double> &);

    void reset();

    void print();

    void randomizeWeights(double, double);
    void mutateWeights(double, double);

    int getSize();

    void setWeight(int, double);
    double getWeight(int);
    void setBias(double);
    double getBias();
    void setNodeIn(double);
    double getNodeIn();
    void setNodeOut(double);
    double getNodeOut();
    void setDelta(double);
    double getDelta();
};


class Layer {
private:
    activationFunctions activation;
    std::vector<Neuron> neurons;
    layerType type;
public:
    Layer();
    Layer(int, int, layerType = HIDDENLAYER, activationFunctions = Sigmoid);

    void reset();

    void print();

    void randomizeWeights(double, double);
    void mutateWeights(double, double);

    void setActivation(activationFunctions);
    activationFunctions getActivation();

    Neuron &getNeuron(int);

    int getSize();

    void setType(layerType);
    layerType getType();
};


#endif //NEURALNETWORK_NEURALNETWORK_H
