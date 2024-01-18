//
// Created by Tobias on 08.01.2024.
//

#include "ImageList.h"

ImageList::ImageList() {

}

void ImageList::addImage(Image &image) {
    this->images.push_back(image);
}

void ImageList::removeImage(int n) {
    images.erase(images.begin() + n);
}

void ImageList::loadImageList(std::string &path) {
    std::ifstream inputFile(path);
    if (!inputFile.is_open()) {
        std::cout << "ERROR: Error opening file!" << "\n";
        return;
    }

    std::string line;
    std::string delimiter = ",";
    bool firstLine = true;

    while (getline(inputFile, line)) {
        if (!firstLine) {
            std::stringstream ss(line);
            std::string token;
            std::vector<double> values;
            bool firstToken = true; // To track the first token for trainingLabel
            Image im;

            while (getline(ss, token, ',')) {


                if (!token.empty()) {
                    if (firstToken) {
                        im.setLabel(token);
                        auto it = std::find(labelDictionary.begin(), labelDictionary.end(), token);
                        if (it == labelDictionary.end()) {
                            labelDictionary.push_back(token);
                        }
                        firstToken = false;
                    } else {
                        double value = stod(token) / 255; // 255 is the highest value possible
                        im.addPixel(value);
                    }
                }
            }
            if (!im.isempty()) {
                this->addImage(im);
            }
        }else{
            firstLine = false;
        }
    }

    inputFile.close();
}

std::string ImageList::getLabel(int n) {
    return images.at(n).getLabel();
}

std::vector<double> &ImageList::getPixels(int n) {
    return images.at(n).getImage();
}

int ImageList::findLabel(std::string label) {
    auto it = find(labelDictionary.begin(), labelDictionary.end(), label);
    return distance(labelDictionary.begin(), it);;
}

bool ImageList::isEmpty() {
    return images.empty();
}

int ImageList::getSize() {
    return images.size();
}

void ImageList::saveImageList(std::string &path) {

}

void ImageList::augmentData(int width, int height, int maxDisplacement){
    for (int i = 0; i < images.size(); ++i) {
        images.at(i).augmentData(width, height, maxDisplacement);
    }
}