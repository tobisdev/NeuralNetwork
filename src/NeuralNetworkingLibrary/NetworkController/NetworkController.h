//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 24.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#ifndef NEURALNETWORK_NETWORKCONTROLLER_H
#define NEURALNETWORK_NETWORKCONTROLLER_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <iomanip>

#include "../ImageList/ImageList.h"
#include "../NeuralNetwork/NeuralNetwork.h"

#define KOKKOS_DEVICES "OpenMP"

class NetworkController {
private:
    NeuralNetwork net;
    ImageList trainingImages;

    std::vector<double> correctSum;
    std::vector<double> errorSum;
    std::vector<double> errorHighest;
    std::vector<double> errorLowest;

    std::vector<double> time;

    double currentError;
public:
    NetworkController(std::vector<int> &, double = 0.25, double = 0.75, activationFunctions = Sigmoid, activationFunctions = Sigmoid, activationFunctions = Sigmoid);

    // Network specific

    void updateNetworkTopology(std::vector<int> &);

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

    void resetNetwork();

    void printNetwork();

    void randomizeWeights(double, double);
    void mutateWeights(double, double);

    int returnGuess();
    std::vector<double> returnOutput();
    void setInput(std::vector<double> &);

    int getNetworkSize();

    double calculateNetworkError(std::vector<double> &desiredValues);

    NeuralNetwork getNetwork();

    // Image Specific

    int getImageSize();
    int getImageListSize();

    void augmentData(int width, int height, int maxDisplacement);

    std::string getLabel();

    // Controller specific

    void loadNetwork(std::string);
    void saveNetwork(std::string);

    void loadTrainingData(std::string);
    void mutateTrainingData();

    void trainNetwork(int, double, double, double, int);

    std::vector<double> &getCorrectSum();
    std::vector<double> &getErrorSum();
    std::vector<double> &getErrorHighest();
    std::vector<double> &getErrorLowest();
    std::vector<double> &getTime();

};


#endif //NEURALNETWORK_NETWORKCONTROLLER_H
