//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 24.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#include "NetworkController.h"

NetworkController::NetworkController(std::vector<int> &topology, double wMin, double wMax, activationFunctions acIn, activationFunctions acHid, activationFunctions acOut) {
    this->net = NeuralNetwork(topology, acIn, acHid, acOut);
    net.randomizeWeights(wMin, wMax);

    std::cout << "NOTE: Network was created!" << "\n";
    std::cout << "Layout:" << "\n";

    for (int i = 0; i < topology.size(); ++i) {
        std::cout << " > " << topology.at(i) << "\n";
    }

    std::cout << "\n";
}

void NetworkController::loadTrainingData(std::string path) {
    trainingImages.loadImageList(path);
    std::cout << "Pixels: " << trainingImages.getPixels(0).size() << "\n";
}

#
void NetworkController::trainNetwork(int epochs, double initialLearnRate, double learnRateDecreaser, double momentum, int imageDisplacement) {
    std::cout << "NOTE: Network is learning!" << "\n";

    for (int i = 0; i < epochs; ++i) {
        std::cout << "NOTE: Startet Epoch " << i << "!" << "\n";
        std::cout << "learnRate: " << initialLearnRate * pow(10, -i * learnRateDecreaser) << "\n\n";

        errorLowest.push_back(0.0);
        errorHighest.push_back(10e12);
        errorLowest.push_back(0.0);
        correctSum.push_back(0);
        errorSum.push_back(0);

        trainingImages.augmentData(28, 28, imageDisplacement);

        #pragma omp parallel for
        for (int j = 0; j < trainingImages.getSize(); ++j)  {
            int pos = trainingImages.findLabel(trainingImages.getLabel(j));

            std::vector<double> desiredOutput(10, 0.0);
            desiredOutput[pos] = 1;

            std::vector<double> im = trainingImages.getPixels(j);

            #pragma omp critical
            {
                net.setInput(trainingImages.getPixels(j));
                net.feedForward();
                net.backPropagation(desiredOutput);
                net.updateValues(initialLearnRate * pow(10, -i * learnRateDecreaser), momentum);
                currentError = net.calculateError(desiredOutput);
                errorSum.at(i) += currentError;

                if(currentError < errorLowest.at(i)){
                    errorLowest.at(i) = currentError;
                }

                if(currentError > errorHighest.at(i)){
                    errorHighest.at(i) = currentError;
                }

                correctSum.at(i) += (net.returnGuess() == pos ? 1 : 0);
            }
        }

        std::cout << "Correct percentage: " << correctSum.at(i) / trainingImages.getSize() * 100 << "%" << "\n";
        std::cout << "Error Value: " << errorSum.at(i) / trainingImages.getSize() << "\n";
    }
}

std::vector<double> &NetworkController::getCorrectSum(){
    return this->correctSum;
}

std::vector<double> &NetworkController::getErrorSum() {
    return this->errorSum;
}

std::vector<double> &NetworkController::getErrorHighest(){
    return this->errorHighest;
}

std::vector<double> &NetworkController::getErrorLowest(){
    return this->errorLowest;
}

std::vector<double> &NetworkController::getTime(){
    return this->time;
}

void NetworkController::updateNetworkTopology(std::vector<int> &top){}

void NetworkController::setInputFunction(activationFunctions ac){
    net.setInputFunction(ac);
}
void NetworkController::setHidddenFunction(activationFunctions ac){
    net.setHidddenFunction(ac);
}
void NetworkController::setOutputFunction(activationFunctions ac){
    net.setOutputFunction(ac);
}
activationFunctions NetworkController::getInputFunction(){
    return net.getInputFunction();
}
activationFunctions NetworkController::getHidddenFunction(){
    return net.getHidddenFunction();
}
activationFunctions NetworkController::getOutputFunction(){
    return net.getOutputFunction();
}

void NetworkController::feedForward(){
    net.feedForward();
}
void NetworkController::backPropagation(std::vector<double> &desiredValues){
    net.backPropagation(desiredValues);
}
void NetworkController::updateValues(double learnRate, double momentum){
    net.updateValues(learnRate, momentum);
}

Layer &NetworkController::getLayer(int i){
    return net.getLayer(i);
}

void NetworkController::resetNetwork(){
    net.reset();
}

void NetworkController::printNetwork(){
    net.print();
}

void NetworkController::randomizeWeights(double min, double max){
    net.randomizeWeights(min, max);
}
void NetworkController::mutateWeights(double min, double max){
    net.mutateWeights(min, max);
}

int NetworkController::returnGuess(){
    return net.returnGuess();
}
std::vector<double> NetworkController::returnOutput(){
    return net.returnOutput();
}
void NetworkController::setInput(std::vector<double> &in){
    net.setInput(in);
}

int NetworkController::getNetworkSize(){
    return net.getSize();
}

double NetworkController::calculateNetworkError(std::vector<double> &desiredValues){
    return net.calculateError(desiredValues);
}

int NetworkController::getImageSize(){
    return this->trainingImages.getPixels(0).size();
}
int NetworkController::getImageListSize(){
    return this->trainingImages.getSize();
}

void NetworkController::saveNetwork(std::string path) {
    std::ofstream outputFile(path);

    if (!outputFile.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return;
    }

    for (int i = 0; i < net.getSize(); ++i) {
        outputFile << net.getLayer(i).getSize();
        if(i != net.getSize() - 1){
            outputFile << ",";
        }
    }
    outputFile << "\n";

    for (int i = 0; i < net.getSize(); ++i) {
        outputFile << (int)net.getLayer(i).getActivation();
        if(i != net.getSize() - 1){
            outputFile << ",";
        }
    }

    outputFile << "\n";

    for (int i = 1; i < net.getSize(); ++i) {
        for (int j = 0; j < net.getLayer(i).getSize(); ++j) {
            outputFile << net.getLayer(i).getNeuron(j).getBias() << ",";
            for (int k = 0; k < net.getLayer(i).getNeuron(j).getSize(); ++k) {
                outputFile << (net.getLayer(i).getNeuron(j).getWeight(k));

                if(k != net.getLayer(i).getNeuron(j).getSize() - 1){
                    outputFile << ",";
                }
            }
            outputFile << "\n";
        }
    }

    outputFile.close();
}

void NetworkController::loadNetwork(std::string path) {
    std::ifstream inputFile(path);
    if (!inputFile.is_open()) {
        std::cout << "ERROR: Error opening file!" << "\n";
        return;
    }

    std::string line;
    std::string delimiter = ",";
    int lineNum = 0;

    std::vector<int> topology;
    std::vector<activationFunctions> activations;

    int currentLayer = 1;
    int currentNeuron = 0;
    int currentWeight = 0;

    while (getline(inputFile, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<double> values;
        bool firstToken = true;
        while (getline(ss, token, ',')) {
            if (!token.empty()) {
                switch (lineNum) {
                    case 0:
                        topology.push_back(std::stoi(token));
                        break;
                    case 1:
                        activations.push_back(activationFunctions(std::stoi(token)));
                        break;
                    default:
                        if(firstToken) {
                            net.getLayer(currentLayer).getNeuron(currentNeuron).setBias(std::stod(token));
                            firstToken = false;
                        }else{
                            net.getLayer(currentLayer).getNeuron(currentNeuron).setWeight(currentWeight,std::stod(token));
                            currentWeight++;
                        }
                        if (currentWeight >= net.getLayer(currentLayer).getNeuron(currentNeuron).getSize()) {
                            currentWeight = 0;
                            currentNeuron++;
                        }
                        if (currentNeuron >= net.getLayer(currentLayer).getSize()) {
                            currentNeuron = 0;
                            currentLayer++;
                        }
                        if (currentLayer >= net.getSize()) {
                            inputFile.close();
                            return;
                        }
                        break;
                }
            }
        }
        switch (lineNum) {
            case 0:
                break;
            case 1:
                if(activations.size() < 2) {
                    return;
                }
                net = NeuralNetwork(topology, activations.at(0), activations.at(1), activations.at(activations.size() - 1));
                break;
            default:
                break;
        }

        lineNum++;
    }

    inputFile.close();
    return;
}

NeuralNetwork NetworkController::getNetwork() {
    return this->net;
}

void NetworkController::augmentData(int width, int height, int maxDisplacement){
    trainingImages.augmentData(width, height, maxDisplacement);
}