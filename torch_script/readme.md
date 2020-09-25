Steps to create a c++ executable for a pytorch code you need the following

1- Download and install CUDA toolkit 10.2
Source: https://developer.nvidia.com/cuda-10.2-download-archive

2- Download and install CUDNN 7.6.5
Source: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
Navigate to your <installpath> directory containing cuDNN.
Unzip the cuDNN package.
cudnn-x.x-windows-x64-v8.x.x.x.zip
or
cudnn-x.x-windows10-x64-v8.x.x.x.zip
Copy the following files into the CUDA Toolkit directory.
Copy <installpath>\cuda\bin\cudnn*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\bin.
Copy <installpath>\cuda\include\cudnn*.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\include.
Copy <installpath>\cuda\lib\x64\cudnn*.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\lib\x64.
Set the following environment variables to point to where cuDNN is located. To access the value of the $(CUDA_PATH) environment variable, perform the following steps:
Open a command prompt from the Start menu.
Type Run and hit Enter.
Issue the control sysdm.cpl command.
Select the Advanced tab at the top of the window.
Click Environment Variables at the bottom of the window.
Ensure the following values are set:
Variable Name: CUDA_PATH
Variable Value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x
Include cudnn.lib in your Visual Studio project.
Open the Visual Studio project and right-click on the project name.
Click Linker > Input > Additional Dependencies.
Add cudnn.lib and click OK.

3- Download LibTorch
Source: https://pytorch.org/
Select LibTorch for windows and c++

4- Install cmake
Source: https://cmake.org/download/

5- Install Visual Studio 2019 (x64)

Next
Source: https://pytorch.org/tutorials/advanced/cpp_export.html#:~:text=Step%201%3A%20Converting%20Your%20PyTorch,by%20the%20Torch%20Script%20compiler.
1- create the python script and generate the saved model

import torch
import torchvision
# An instance of your model.
model = torchvision.models.resnet18()
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet_model.pt")

2- create the C++ code for input and model import
#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
}

3- create the CMAKE list file
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)


Build
1- Arrange the files in the following hierarchy
example-app/
  CMakeLists.txt
  example-app.cpp

2- run cmake with the LibTorch path
Source: https://pytorch.org/tutorials/advanced/cpp_export.html#:~:text=Step%201%3A%20Converting%20Your%20PyTorch,by%20the%20Torch%20Script%20compiler.
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release

4- test the executable
