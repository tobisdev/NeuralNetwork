//
// Created by Tobias on 08.01.2024.
//

#ifndef NEURALNETWORK_IMAGE_H
#define NEURALNETWORK_IMAGE_H

#include <iostream>
#include <vector>

class Image {
private:
    std::string label;
    std::vector<double> pixels;
public:
    Image();
    Image(std::vector<double> &, std::string &);

    void augmentData(int width, int height, int maxDisplacement);

    void addPixel(double);
    void removePixel(int);

    bool isempty();
    int getSize();

    void setImage(std::vector<double> &);
    std::vector<double> &getImage();
    void setLabel(std::string &);
    std::string getLabel();
};


#endif //NEURALNETWORK_IMAGE_H
