

<picture>
  <img src="syn0134_deformation.gif" width="350" title="bone deformation during screw push-out" align="right">
  <img alt="BoneDeforming">
</picture>

# MBS-3D-OptFlow

MBS-3D-OptFlow provides fast and memory efficient digital volume correlation for Nvidia GPUs intended for the use with synchrotron µ-CT volume image data.

**Pending Updates**:
  - Add a tool to perform rigid_body registration
  - Documentation of voxel2mesh
  - Streamlining of main.cpp (removing unused experimental features, workflows to separate files)
  - Documenting all available input parameters
<br>

## References
The code is a further development of 2D code provided by A. Ershov: ...
<br>
Optimization of the energy functional is performed with successive overrelaxation as introduced in 2D by Liu: ...
<br>
Default derivatives are calculated according to Farid and Simoncelli: ...
<br>
Not to forget the fundamental works of Brox and Weickert: ...

<br>

## Quick Start Guide

### Compilation

Required libraries are LibTiff and OpenMP. The source code should compile on most Linux distributions by providing the **location of your nvcc compiler** and the **CUDA compute capability** of your GPU in the script file *build_mbsoptflow.sh*. Execute with 

***<p align="center"> bash build_mbsoptflow.sh </p>***

which will provide an executable *mbsoptflow* in the same directory.
<br>
<br> 
- With the switch **BUILD_STRAINMAPPER=true** the program ***voxel2mesh*** will also be build for analysis. ***voxel2mesh*** provides some basic postprocessing functionalities and projects the DVC results on a medit- or vtk-mesh.
- With the switch **UNZIP_DEMOS=true** small demo projects will be extracted to the ***Demos*** subdirectory.
<br>

### Performing DVC

It is highly recommended that you perform denoising and a rigid body registration (or at least a coarse manual registration of datasets) before calculating displacement fields with the DVC algorithm. Preregistering the data assures a large field of view and minimizes displacements. Larger motions across image boundaries and motions that are large in relation to the moving object may still pose problems.

You need to provide two greyscale tif-image sequences (8bit, 16bit or 32bit): a reference (Frame0) and a transformed image sequence (Frame1). 3D-tif files are also supported but may result in unexpected behavior. Both image sequences need to be of the same height, widht and depth. The default output will be a dense field of displacement vectors. Frame0 and Frame1 are passed with the -i0 and -i1 argument. An output directory may be specified with the -o argument. Thus, the most basic program call would be:

***<p align="center"> ./mbsoptflow -i0 /path/to/my/reference/data/ -i1 /path/to/my/displaced/data/ -o /path/to/dvc/output/</p>***

A full list of available arguments and flags will follow. Right now you need to check the section *extract command line arguments* in *main.cpp* and/or *protocol_parameters.h*. The latter sets the default parameters  used during compilation that can be overwritten by the arguments.

<br>

### Basic Analysis

The supporting programm voxel2mesh allows mapping the displacement vectors calculated with DVC on a Medit mesh (*.mesh) or VTK mesh (*.vtk). Documentation will be provided.

<br>

### Preprocessing

It is recomended to reduce noise and artefacts as good as possible. Furthermore, preregistering with a rigid body transformation maximizes the field of view and minimizes larger motions across the image boundaries. A basic tool will follow.

<br>

## Demos 
[Basic Functionality](https://github.com/brunsst/MBS-3D-OptFlow/blob/main/Demos/RayDemo/README.md)
<br>
[Tensile Testing with Little Textures](https://github.com/brunsst/MBS-3D-OptFlow/blob/main/Demos/WireDemo/README.md)

<br>

## Exemplary Program Calls

[Example Call 1: Cracks and strain in bone (large dataset, localglobal method)](https://github.com/brunsst/MBS-3D-OptFlow/blob/main/examplary_call1.md)
<br>
[Example Call 2: Tensile tests on SMA wires (motion across image boundary, little textures)](https://github.com/brunsst/MBS-3D-OptFlow/blob/main/examplary_call2.md)
<br>
[Example Call 3: Rearranged sand grains (warping, very large dataset)](https://github.com/brunsst/MBS-3D-OptFlow/blob/main/examplary_call3.md)

<br>

## Publications
When using the code please cite:...
The code was also used in:...
<br>

## Acknowledgements
The project was performed at... 
Funding was received from...

