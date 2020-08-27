The example-add app is based on https://pytorch.org/tutorials/advanced/cpp_export.html

First, get libtorch:

    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
    
    unzip libtorch-shared-with-deps-latest.zip




We can now run the following commands to build the application from within the example-app/ folder:

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release