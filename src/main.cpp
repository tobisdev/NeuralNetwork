//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 14.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#include <filesystem>

#include "NeuralNetworkingLibrary/include.h"

#include <thread>

#include <omp.h>


std::vector<int> topology = {784, 10};
NetworkController master(topology, 0.25, 0.55, LeakyReLU, LeakyReLU, LeakyReLU);

void train(){
    std::string toReplace = "cmake-build-debug";
    std::stringstream replaceWith;
    replaceWith << "miscellaneous" << (char) 92 << "TrainingData" << (char) 92 << "train.csv";

    std::string path = std::filesystem::current_path().string();
    std::string finalPath = path.replace(path.find(toReplace), toReplace.length(), replaceWith.str());

    master.loadTrainingData(path);

    master.trainNetwork(20, 1e-4, 0.03, 0.01, 0);
    master.saveNetwork("net.csv");
}

int main() {
    omp_set_num_threads(4);

    master.loadNetwork("net.csv");

    for (int i = 0; i < master.getNetworkSize(); ++i) {
        std::cout << master.getLayer(i).getSize() << "\n";
    }

    std::thread myThread(train);

    GUI myGui(master);
    myGui.init();
    myGui.startRenderLoop();
    myGui.shutdown();
}
