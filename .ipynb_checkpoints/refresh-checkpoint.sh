#!/bin/bash

# Copy files
cp kernel.cu ../
cp pytorch_apis.py ../

# Run Python script
python3 main.py

# Copy files back
cp ../kernel.cu .
cd build/
cmake ../
make
cp graphpy.cpython-37m-x86_64-linux-gnu.so ../
cd ..
