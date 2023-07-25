#!/bin/bash

############# Parameters ################
#########################################
CUDA_COMPUTECAPABILITY=70
NVCC_PATH="/usr/local/cuda-10.0/bin/nvcc"

#########################################
#########################################

mkdir -p "Build"

function cpu_compile()
{
    filename=$1
    subdir=$2
    echo "compiling "$filename".cpp"
    $NVCC_PATH -O3 -w -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o   "Build/"$filename".d" $subdir"/"$filename".cpp"
    $NVCC_PATH -O3 -w -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/"$filename".o" $subdir"/"$filename".cpp"
}
function cuda_compile()
{
    filename=$1
    echo "compiling "$filename".cu"
    $NVCC_PATH --disable-warnings -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/"$filename".d" $filename".cu"
    $NVCC_PATH --disable-warnings -O3 -Xcompiler -fopenmp -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=compute_$CUDA_COMPUTECAPABILITY -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -x cu -o  "Build/"$filename".o" $filename".cu"
}

if test $CUDA_COMPUTECAPABILITY != 0 && test $CUDA_COMPUTECAPABILITY != 00
then
    echo "building cudaWBBOptFlow_v0.2"
    echo "--------------------------------------------------"
    echo "compiling main.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o   "Build/main.d" "main.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/main.o" "main.cpp"

    cuda_compile "registration_bruteforce"
    cpu_compile "auxiliary" "Geometry"
    cpu_compile "hdcommunication" "Geometry"
    cpu_compile "histogram" "Geometry"

    echo "linking"
    $NVCC_PATH --cudart static --relocatable-device-code=false -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=compute_$CUDA_COMPUTECAPABILITY -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY -link -o  "wbbregistration"  ./Build/main.o  ./Build/registration_bruteforce.o ./Build/auxiliary.o ./Build/hdcommunication.o ./Build/histogram.o -ltiff -lgomp
    echo "--------------------------------------------------"
else
    echo "Cannot build without CUDA support!"
fi

################################################################################################################################################################
################################################################################################################################################################
