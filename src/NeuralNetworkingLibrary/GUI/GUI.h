//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 14.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#ifndef NEURALNETWORK_GUI_H
#define NEURALNETWORK_GUI_H

#include <SFML/Graphics.hpp>
#include "imgui.h"
#include "imgui-SFML.h"
#include "../NetworkController/NetworkController.h"

#include "Addons/implot/implot.h"
#include "Addons/NetworkViewer/NetworkViewer.h"
#include "Addons/DrawingApplication/DrawingApplication.h"


class GUI{
private:
    sf::RenderWindow window;
    NetworkController &controller;
    void testImage();
public:
    GUI(NetworkController &);
    void init();
    void startRenderLoop();
    void update();
    void shutdown();
};

#endif //NEURALNETWORK_GUI_H
