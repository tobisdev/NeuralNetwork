//################################//
//                                //
// Neural Network by Huber Tobias //
//  Date of Creation 14.12.2023   //
//  © 2023 License: GNU AGPLv3    //
//                                //
//################################//

#include "GUI.h"

// Mouse
static bool ableToPress = false;
static bool isMouseReleased = false;

// Global Variables
// Drawing Application
int gridSize = 28;
std::vector<std::vector<double>> buttonStates(gridSize, std::vector<double>(gridSize, 0));

// Settings Menu
std::string openNetPath;
std::string openDatPath;
char saveNetPath[255];


GUI::GUI(NetworkController &controller) : window(sf::VideoMode(800, 600), "Neural Network"), controller(controller) {

}

void GUI::init() {
// ImGui initialization for SFML
    ImGui::SFML::Init(window);

    ImGui::CreateContext();

    // ImGui context initialization
    IMGUI_CHECKVERSION();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Setup ImGui style
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowRounding = 12;
    style.FrameRounding = 12;
    style.TabRounding = 12;
}

void GUI::update() {
    ImGui::SFML::Update(window, sf::seconds(1.f / 360.f));

    // Start the ImGui dock space
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    // Create a dockable window
    if (ImGui::Begin("Neural Network Data")) {

        std::vector<double> myCorrectSum;
        std::vector<double> myErrorSum;

        myErrorSum.push_back(100);
        myCorrectSum.push_back(0);

        int maxCorrect = 0;
        int maxError = 0;

        for (int i = 0; i < controller.getCorrectSum().size(); ++i) {
            double myCorrectValue = controller.getCorrectSum().at(i) / controller.getImageListSize() * 100;
            double myErrorValue = controller.getErrorSum().at(i) / controller.getImageListSize();

            myCorrectSum.push_back(myCorrectValue);
            myErrorSum.push_back(myErrorValue);

            if (myCorrectValue > maxCorrect) {
                maxCorrect = myCorrectValue;
            }
            if (myErrorValue > maxError) {
                maxError = myErrorValue;
            }
        }

        ImPlot::CreateContext();
        if (ImPlot::BeginPlot("Neural Network Learning Progress")) {

            ImPlot::SetupAxisLimits(ImAxis_X1, 0, myErrorSum.size() - 1);
            ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 100);

            ImPlot::PlotLine("Average Error", myErrorSum.data(), myErrorSum.size());
            ImPlot::PlotLine("Correct  [%]", myCorrectSum.data(), myCorrectSum.size());

            ImPlot::EndPlot();
        }
        ImPlot::DestroyContext();

        std::string out = "Images: ";
        out += controller.getImageListSize();

        ImGui::Text(out.c_str());
    }
    ImGui::End();

    float boxSize = 10000 / ImGui::GetContentRegionAvail().x;

    if (ImGui::Begin("Neural Network Weight Viewer")) {
        NetworkViewer::viewNetwork(controller, boxSize, 1, 1, 0.4);
    }
    ImGui::End();

    ImGui::ShowDemoWindow();

    if (ImGui::Begin("Drawing Application")) {
        DrawingApplication::paintingTool(buttonStates, gridSize, 0.2, controller);
        if(ImGui::IsKeyDown(ImGuiKey_C)){
            for (int i = 0; i < buttonStates.size(); ++i) {
                for (int j = 0; j < buttonStates[i].size(); ++j) {
                    buttonStates[i][j] = 0;
                }
            }
        }
        if(ableToPress){
            if(ImGui::IsMouseDown(ImGuiMouseButton_Left)){
                isMouseReleased = false;
                ableToPress = false;
            }
            if(isMouseReleased){
                isMouseReleased = false;
                testImage();
            }
        }else{
            if(!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                isMouseReleased = true;
                ableToPress = true;
            }
        }
    }

    ImGui::End();


    if (ImGui::Begin("Main Menu")) {
        if (ImGui::CollapsingHeader("Network Settings")){
            // Render the contents inside the collapsible header
            ImGui::SeparatorText("Save Neural Network");
            // Render ImGui elements
            // Render ImGui elements
            ImGui::InputText("##", saveNetPath, sizeof(saveNetPath));
            ImGui::SameLine();
            if(ImGui::Button("...")){
                std::cout << "Button";
            }
            ImGui::NewLine();
            if(ImGui::Button("Load Network")){
                std::cout << "Button";
            }

            ImGui::SeparatorText("Load Neural Network");


            ImGui::SeparatorText("Neural Network Settings");

        }
    }
    ImGui::End();

    if (ImGui::Begin("Actions Menu")) {

    }
    ImGui::End();
}

void GUI::startRenderLoop() {
    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            ImGui::SFML::ProcessEvent(event);

            if (event.type == sf::Event::Closed){
                window.close();
                break;
            }
        }

        update();

        window.clear();
        ImGui::SFML::Render(window);
        window.display();
    }

    GUI::shutdown();
}

void GUI::shutdown() {
    ImGui::DestroyContext();
    ImGui::SFML::Shutdown();
}

void GUI::testImage(){
    NeuralNetwork testNet = controller.getNetwork();
    std::vector<double> input;

    for (int i = 0; i < buttonStates.size(); ++i) {
        for (int j = 0; j < buttonStates[i].size(); ++j) {
            input.push_back(buttonStates[i][j]);
        }
    }

    testNet.setInput(input);
    testNet.feedForward();
    int outIdx = testNet.returnGuess();

    std::cout << "I guess: '" << outIdx << "\n";
}