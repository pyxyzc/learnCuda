#!/usr/bin/bash

export MY_INCLUDE_PATH="\
/usr/local/lib/python3.12/dist-packages/torch/include:\
/usr/local/lib/python3.12/dist-packages/torch/include/torch/csrc/api/include:\
/usr/include/python3.12/:\
/usr/local/cuda/include"

if [[ -d ./build ]]; then
	rm -rf ./build
fi

cmake -S . -B build \
	-DCMAKE_BUILD_TYPE=Release \
	-DRUNTIME_ENVIRONMENT=cuda \
	-DPYTHON_EXECUTABLE=/usr/local/lib/python3.12/ \
	-Dpybind11_DIR=/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11 \
	-DTorch_DIR=/usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch \
	-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

cd build
make
cd ..
