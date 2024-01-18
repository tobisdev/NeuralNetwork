//
// Created by Tobias on 08.01.2024.
//

#include "Image.h"

Image::Image() {

}

void Image::addPixel(double a) {
    this->pixels.push_back(a);
}

void Image::removePixel(int n) {
    pixels.erase(pixels.begin() + n);
}

Image::Image(std::vector<double> &pixels, std::string &label) {
    this->pixels = pixels;
    this->label = label;
}

void Image::setImage(std::vector<double> &pixels) {
    this->pixels = pixels;
}

std::vector<double> &Image::getImage() {
    return this->pixels;
}

void Image::setLabel(std::string &label) {
    this->label = label;
}

std::string Image::getLabel() {
    return this->label;
}

bool Image::isempty() {
    return this->label.empty();
}

int Image::getSize() {
    return this->pixels.size();
}

void Image::augmentData(int width, int height, int maxDisplacement) {
    int displacement_x = rand() % (2 * maxDisplacement + 1) - maxDisplacement;
    int displacement_y = rand() % (2 * maxDisplacement + 1) - maxDisplacement;

    std::vector<double> augmentedData(pixels.size(), 0.0);
    for (int i = 0; i < pixels.size(); ++i) {
        int original_x = i % width;
        int original_y = i / width;

        int new_x = (original_x + displacement_x + width) % width;
        int new_y = (original_y + displacement_y + height) % height;

        int new_index = new_y * width + new_x;
        augmentedData[new_index] = pixels[i];
    }
    pixels = augmentedData;
}