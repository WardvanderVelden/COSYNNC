cmake_minimum_required(VERSION 3.16)
project(COSYNNC)

# Require C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Allow for multithreading
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

# Set the path to where MXNET is located
set(MXNET_PATH /mxnet/path)


# Add MXNet include folders
include_directories(${MXNET_PATH}/cpp-package/include)
include_directories(${MXNET_PATH}/include)
include_directories(${MXNET_PATH}/3rdparty/dmlc-core/include)
include_directories(${MXNET_PATH}/3rdparty/tvm/nnvm/include)

# Add COSYNNC include folders
include_directories(src)
include_directories(src/control)
include_directories(src/interface)
include_directories(src/neuralnetworks)
include_directories(src/plants)
include_directories(src/utilities)

# Add files that make up the executable
add_executable(COSYNNC
        src/control/Abstraction.cpp
        src/control/Abstraction.h
        src/control/Controller.cpp
        src/control/Controller.h
        src/control/ControlSpecification.cpp
        src/control/ControlSpecification.h
        src/control/Quantizer.cpp
        src/control/Quantizer.h
        src/control/Transition.cpp
        src/control/Transition.h
        src/control/Verifier.cpp
        src/control/Verifier.h
        src/interface/BddManager.cpp
        src/interface/BddManager.h
        src/interface/FileManager.cpp
        src/interface/FileManager.h
        src/neuralnetworks/MultilayerPerceptron.cpp
        src/neuralnetworks/MultilayerPerceptron.h
        src/neuralnetworks/NeuralNetwork.cpp
        src/neuralnetworks/NeuralNetwork.h
        src/plants/DCDC.cpp
        src/plants/DCDC.h
        src/plants/LinearHybrid.cpp
        src/plants/LinearHybrid.h
        src/plants/Plant.cpp
        src/plants/Plant.h
        src/plants/Rocket.cpp
        src/plants/Rocket.h
        src/plants/StateSpaceRepresentation.cpp
        src/plants/StateSpaceRepresentation.h
        src/plants/Unicycle.cpp
        src/plants/Unicycle.h
        src/utilities/Edge.cpp
        src/utilities/Edge.h
        src/utilities/Hyperplane.cpp
        src/utilities/Hyperplane.h
        src/utilities/StringHelper.cpp
        src/utilities/StringHelper.h
        src/utilities/Vector.cpp
        src/utilities/Vector.h
        src/Procedure.cpp
        src/Procedure.h
        src/Encoder.cpp
        src/Encoder.h
        src/main.cpp)

# Link the required libraries
target_link_libraries(COSYNNC ${MXNET_PATH}/build/libmxnet.so)
