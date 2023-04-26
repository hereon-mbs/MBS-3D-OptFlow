# MBS-3D-OptFlow

MBS-3D-OptFlow provides fast and memory efficient digital volume correlation for Nvidia GPUs.

**26.04.2023:** Dumped the source code for knowledgeable user.

**Pending Updates**:
  - Streamlining of main.cpp (removing unused experimental features, workflows to separate files)
  - Documenting all available input parameters
  - Upload demo projects

<br>

## Quick Start Guide

### Compilation

Required libraries are LibTiff and OpenMP. The source code should compile on most Linux distributions by providing the **location of your nvcc compiler** and the **CUDA compute capability** of your GPU in the script file *build_mbsoptflow.sh*. Execute with 

***<p align="center"> sh build_mbsoptflow.sh </p>***

which will provide an executable *mbsoptflow* in the same directory.

<br>

### Performing DVC




## Exemplary Program Calls
