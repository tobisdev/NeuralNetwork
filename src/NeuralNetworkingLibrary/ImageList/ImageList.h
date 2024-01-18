//
// Created by Tobias on 08.01.2024.
//

#ifndef NEURALNETWORK_IMAGELIST_H
#define NEURALNETWORK_IMAGELIST_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <vector>

#include "Image/image.h"

class ImageList {
private:
    std::vector<Image> images;
    std::vector<std::string> labelDictionary;
public:
    ImageList();

    void augmentData(int width, int height, int maxDisplacement);

    int findLabel(std::string);

    std::vector<double> &getPixels(int);
    std::string getLabel(int);

    void addImage(Image &);
    void removeImage(int);

    bool isEmpty();
    int getSize();

    void loadImageList(std::string &);
    void saveImageList(std::string &);
};

#endif //NEURALNETWORK_IMAGELIST_H
