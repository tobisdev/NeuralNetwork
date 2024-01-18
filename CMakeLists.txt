cmake_minimum_required(VERSION 3.26)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 17)

# Find SFML
set(SFML_DIR "miscellaneous/SFML/lib/cmake/SFML")
find_package(SFML 2.5 COMPONENTS system window graphics REQUIRED)

# Add ImGui and ImGui-SFML
include_directories(miscellaneous/DearImgui)
include_directories(miscellaneous/ImGui-SFML)

file(GLOB IMGUI_SOURCES "miscellaneous/DearImgui/*.cpp" "miscellaneous/DearImgui/*.h")
file(GLOB IMGUI_SFML_SOURCES "miscellaneous/ImGui-SFML/*.cpp" "miscellaneous/ImGui-SFML/*.h")

# Glob all .cpp and .h files in src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.h")

# Add the source files to your project
add_executable(NeuralNetwork ${SOURCES} ${IMGUI_SOURCES} ${IMGUI_SFML_SOURCES})

# Find OperMP and link it to the project
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NeuralNetwork PUBLIC OpenMP::OpenMP_CXX -lgomp)
endif()

# Link SFML and OpenGL libraries
target_link_libraries(NeuralNetwork PUBLIC sfml-system sfml-window sfml-graphics)
find_package(OpenGL REQUIRED)
target_link_libraries(NeuralNetwork PUBLIC OpenGL::GL)

# Copy SFML DLLs to the output directory
set(SFML_DLLS "${SFML_DIR}/../../../bin")
file(GLOB SFML_DLL_FILES "${SFML_DLLS}/*.dll")
foreach(DLL ${SFML_DLL_FILES})
    add_custom_command(TARGET NeuralNetwork POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DLL}" $<TARGET_FILE_DIR:NeuralNetwork>)
endforeach()

# Compile with OpenMP
target_compile_options(NeuralNetwork PRIVATE -fopenmp)