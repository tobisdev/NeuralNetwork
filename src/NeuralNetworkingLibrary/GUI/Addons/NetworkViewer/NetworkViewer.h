//
// Created by Tobias on 14.01.2024.
//

#ifndef NEURALNETWORK_NETWORKVIEWER_H
#define NEURALNETWORK_NETWORKVIEWER_H

class NetworkViewer{
public:
    static void viewNetwork(NetworkController &controller, int boxSize, double cx, double cy, double cz){
        ImGuiStyle &style = ImGui::GetStyle();
        int rounding = style.FrameRounding;
        style.FrameRounding = 0;
        ImVec2 defaultSpacing = style.ItemSpacing;
        style.ItemSpacing = ImVec2(0.0f, 0.0f);

        for (int i = 0; i < controller.getNetworkSize(); ++i) {
            Layer &layer = controller.getLayer(i);
            for (int j = 0; j < layer.getSize(); ++j) {
                Neuron &neuron = layer.getNeuron(j);
                // Map weight value to color
                ImVec4 color = ImVec4(0.5 - neuron.getNodeIn() * cx, neuron.getNodeOut() * cy,
                                      0.5 + neuron.getNodeOut() * cz, 1);

                // Draw the colored box
                ImGui::PushStyleColor(ImGuiCol_Button, color);

                ImGui::Button("##", ImVec2(boxSize, boxSize));
                ImGui::PopStyleColor();

                // Tooltip with weight value
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::Text("Layer: %d\nNeuron: %d\nInput Value: %.4f\nOutput Value: %.4f", i, j,
                                neuron.getNodeIn(), neuron.getNodeOut());
                    ImGui::EndTooltip();
                }

                // Move the cursor to the next position
                ImGui::SameLine();
                if (ImGui::GetCursorPosX() + boxSize >= ImGui::GetWindowWidth()) {
                    // Move to the next line
                    ImGui::NewLine();
                }
            }
        }
        style.ItemSpacing = defaultSpacing;
        style.FrameRounding = rounding;
    }
};

#endif //NEURALNETWORK_NETWORKVIEWER_H
