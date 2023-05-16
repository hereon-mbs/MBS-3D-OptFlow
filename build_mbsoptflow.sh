#!/bin/bash

############# Parameters ################
#########################################
CUDA_COMPUTECAPABILITY=70
#CUDA_COMPUTECAPABILITY=70
NVCC_PATH="/usr/local/cuda-10.0/bin/nvcc"

#########################################
#########################################

mkdir -p "Build"

function cpu_compile()
{
    filename=$1
    subdir=$2
    echo "compiling "$filename".cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o   "Build/"$filename".d" $subdir"/"$filename".cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/"$filename".o" $subdir"/"$filename".cpp"
}
function cuda_compile()
{
    filename=$1
    subdir=$2
    echo "compiling "$filename".cu"
    $NVCC_PATH --disable-warnings -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o "Build/"$filename".d" $subdir"/"$filename".cu"
    $NVCC_PATH --disable-warnings -O3 -Xcompiler -fopenmp -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=compute_$CUDA_COMPUTECAPABILITY -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -x cu -o  "Build/"$filename".o" $subdir"/"$filename".cu"
}

if test $CUDA_COMPUTECAPABILITY != 0 && test $CUDA_COMPUTECAPABILITY != 00
then
    echo "building cudaWBBOptFlow_v0.2"
    echo "--------------------------------------------------"
    echo "compiling main.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY  -odir "Build" -M -o   "Build/main.d" "Source/main.cpp"
    $NVCC_PATH -O3 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "Build/main.o" "Source/main.cpp"

    cpu_compile "farid_kernels" "Source/Solver"
    cpu_compile "optflow_cpu2d" "Source/Solver"
    cpu_compile "optflow_cpu3d" "Source/Solver"
    cuda_compile "optflow_gpu2d" "Source/Solver"
    cuda_compile "optflow_gpu2d_reshape" "Source/Solver"
    cuda_compile "optflow_gpu3d" "Source/Solver"
    cuda_compile "optflow_gpu3d_reshape" "Source/Solver"
    cuda_compile "register_correlationwindow" "Source/Preprocessing"
    cpu_compile "resampling" "Source/Scaling"
    cpu_compile "pyramid" "Source/Scaling"
    cpu_compile "resampling" "Source/Scaling"
    cpu_compile "histomatching" "Source/Preprocessing"
    cpu_compile "cornerdetection_cpu" "Source/Preprocessing"
    cpu_compile "auxiliary" "Source/Geometry"
    cpu_compile "derivatives" "Source/Geometry"
    cpu_compile "eig3" "Source/Geometry"
    cpu_compile "filtering" "Source/Geometry"
    cpu_compile "hdcommunication" "Source/Geometry"
    cpu_compile "histogram" "Source/Geometry"
    cpu_compile "smoothnessterm_cpu" "Source/Derivatives"  

    echo "linking"
    $NVCC_PATH --cudart static --relocatable-device-code=false -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=compute_$CUDA_COMPUTECAPABILITY -gencode arch=compute_$CUDA_COMPUTECAPABILITY,code=sm_$CUDA_COMPUTECAPABILITY -link -o  "mbsoptflow"  ./Build/main.o  ./Build/farid_kernels.o ./Build/optflow_cpu2d.o ./Build/optflow_cpu3d.o ./Build/optflow_gpu2d.o ./Build/optflow_gpu2d_reshape.o ./Build/optflow_gpu3d.o ./Build/optflow_gpu3d_reshape.o  ./Build/pyramid.o ./Build/resampling.o  ./Build/histomatching.o  ./Build/cornerdetection_cpu.o  ./Build/auxiliary.o ./Build/derivatives.o ./Build/eig3.o ./Build/filtering.o ./Build/hdcommunication.o ./Build/histogram.o  ./Build/register_correlationwindow.o ./Build/smoothnessterm_cpu.o   -ltiff -lgomp
    echo "--------------------------------------------------"
else
    echo "Cannot build without CUDA support!"
fi

################################################################################################################################################################
################################################################################################################################################################
