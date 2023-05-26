#!/bin/bash

############# Parameters ################
#########################################
CUDA_COMPUTECAPABILITY=70
NVCC_PATH="/usr/local/cuda-10.0/bin/nvcc"

BUILD_BINARY=true
BUILD_STRAINMAPPER=true
UNZIP_DEMOS=true
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

if $BUILD_BINARY && test $CUDA_COMPUTECAPABILITY != 0 && test $CUDA_COMPUTECAPABILITY != 00
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
    if [ ! $BUILD_BINARY ]
    then
        echo "Cannot build without CUDA support!"
    fi
fi

if $BUILD_STRAINMAPPER
then
    echo "building voxel2mesh mapper"
    echo "--------------------------------------------------"
    echo "compiling displacement_derivatives.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Derivatives/displacement_derivatives.cpp" -o $PWD"/Postprocessing/Build/displacement_derivatives.o"
    echo "compiling polar_decomposition.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Derivatives/polar_decomposition.cpp" -o $PWD"/Postprocessing/Build/polar_decomposition.o"
    echo "compiling meshio.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Geometry/meshio.cpp" -o $PWD"/Postprocessing/Build/meshio.o"
    echo "compiling auxiliary.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Geometry/auxiliary.cpp" -o $PWD"/Postprocessing/Build/auxiliary.o"
    echo "compiling hdcommunication.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Geometry/hdcommunication.cpp" -o $PWD"/Postprocessing/Build/hdcommunication.o"
    echo "compiling filtering.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Geometry/filtering.cpp" -o $PWD"/Postprocessing/Build/filtering.o"
    echo "compiling taubin_smoothing.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Geometry/taubin_smoothing.cpp" -o $PWD"/Postprocessing/Build/taubin_smoothing.o"
    echo "compiling triangles.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/Geometry/triangles.cpp" -o $PWD"/Postprocessing/Build/triangles.o"
    echo "compiling main.cpp"
    g++ -w -fexceptions -O3 -std=c++11 -fopenmp  -c $PWD"/Postprocessing/main.cpp" -o $PWD"/Postprocessing/Build/main.o"
    echo "linking voxel2mesh"
    g++  -o $PWD/voxel2mesh $PWD/Postprocessing/Build/main.o  $PWD/Postprocessing/Build/auxiliary.o $PWD/Postprocessing/Build/hdcommunication.o $PWD/Postprocessing/Build/displacement_derivatives.o $PWD/Postprocessing/Build/polar_decomposition.o $PWD/Postprocessing/Build/meshio.o $PWD/Postprocessing/Build/filtering.o $PWD/Postprocessing/Build/taubin_smoothing.o $PWD/Postprocessing/Build/triangles.o -ltiff -lgomp
    echo "--------------------------------------------------"
fi

if $UNZIP_DEMOS
then
    echo "unzipping ray demo..."
    unzip -qo Demos/RayDemo/Frame01.zip -d Demos/RayDemo/
    unzip -qo Demos/RayDemo/Frame02.zip -d Demos/RayDemo/
    unzip -qo Demos/RayDemo/Frame03.zip -d Demos/RayDemo/
    unzip -qo Demos/RayDemo/Mask01.zip -d Demos/RayDemo/
    echo "--------------------------------------------------"
fi

################################################################################################################################################################
################################################################################################################################################################
