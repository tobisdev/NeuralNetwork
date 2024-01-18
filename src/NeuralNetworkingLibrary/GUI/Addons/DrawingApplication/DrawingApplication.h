//
// Created by Tobias on 14.01.2024.
//

#ifndef NEURALNETWORK_DRAWINGAPPLICATION_H
#define NEURALNETWORK_DRAWINGAPPLICATION_H

class DrawingApplication{
private:

public:
    static void paintingTool(std::vector<std::vector<double>> &buttonStates, int gridSize, double brushStrength, NetworkController &controller) {
        ImVec2 squareSize = ImVec2(ImGui::GetContentRegionAvail().x / gridSize, ImGui::GetContentRegionAvail().x / gridSize);

        if (ImGui::GetContentRegionAvail().x > ImGui::GetContentRegionAvail().y) {
            squareSize = ImVec2(ImGui::GetContentRegionAvail().y / gridSize, ImGui::GetContentRegionAvail().y / gridSize);
        }

        int startX = ImGui::GetWindowPos().x + ImGui::GetContentRegionAvail().x / 2 - squareSize.x * gridSize / 2 + 8;
        int startY = ImGui::GetWindowPos().y + ImGui::GetContentRegionAvail().y / 2 - squareSize.y * gridSize / 2 + 25;

        int backroundColor = 51;
        int colorDraw;

        // Draw colored squares in a grid
        for (int i = 0; i < gridSize; ++i) {
            for (int j = 0; j < gridSize; ++j) {
                colorDraw = buttonStates[i][j] * 255 > backroundColor ? buttonStates[i][j] * 255 : backroundColor;
                ImU32 color = IM_COL32(colorDraw, colorDraw, colorDraw, 255);
                ImVec2 topLeft(startX + j * squareSize.x, startY + i * squareSize.y);
                ImVec2 bottomRight(startX + (j + 1) * squareSize.x + 1, startY + (i + 1) * squareSize.y + 1);

                ImGui::GetWindowDrawList()->AddRectFilled(topLeft, bottomRight, color);

                ImVec2 mousePos = ImGui::GetMousePos();

                if (mousePos.x > topLeft.x && mousePos.y > topLeft.y && mousePos.x < bottomRight.x && mousePos.y < bottomRight.y && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                    // Toggle the square state on click
                    if (buttonStates[i][j] + brushStrength < 1) {
                        buttonStates[i][j] += brushStrength;
                    } else {
                        buttonStates[i][j] = 1;
                    }
                }

                if (mousePos.x > topLeft.x && mousePos.y > topLeft.y && mousePos.x < bottomRight.x && mousePos.y < bottomRight.y && ImGui::IsMouseDown(ImGuiMouseButton_Right)) {
                    // Toggle the square state on click
                    if (buttonStates[i][j] - brushStrength > 0) {
                        buttonStates[i][j] -= brushStrength;
                    } else {
                        buttonStates[i][j] = 0;
                    }
                }
            }
        }
    }
};

#endif //NEURALNETWORK_DRAWINGAPPLICATION_H
