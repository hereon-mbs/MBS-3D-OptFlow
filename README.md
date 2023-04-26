# MBS-3D-OptFlow

MBS-3D-OptFlow provides fast and memory efficient digital volume correlation for Nvidia GPUs intended for the use with synchrotron µ-CT volume image data.

**26.04.2023:** Dumped the source code for knowledgeable users.

**Pending Updates**:
  - Streamlining of main.cpp (removing unused experimental features, workflows to separate files)
  - Documenting all available input parameters
  - Upload demo projects (ray cartiledge; NiTi wires)

<br>

## Quick Start Guide

### Compilation

Required libraries are LibTiff and OpenMP. The source code should compile on most Linux distributions by providing the **location of your nvcc compiler** and the **CUDA compute capability** of your GPU in the script file *build_mbsoptflow.sh*. Execute with 

***<p align="center"> sh build_mbsoptflow.sh </p>***

which will provide an executable *mbsoptflow* in the same directory.

<br>

### Performing DVC

It is highly recommended that you perform denoising and a rigid body registration (or at least a coarse manual registration of datasets) before calculating displacement fields with the DVC algorithm. Preregistering the data assures a large field of view and minimizes displacements. Larger motions across image boundaries and motions that are large in relation to the moving object may still pose problems.

You need to provide two greyscale tif-image sequences (8bit, 16bit or 32bit): a reference (Frame0) and a transformed image sequence (Frame1). 3D-tif files are also supported. Both image sequences need to be of the same height, widht and depth. The default output will be a dense field of displacement vectors. Frame0 and Frame1 are passed with the -i0 and -i1 argument. Currently there is no default output directory (will change that and at least catch exceptions from I/O). You need to provide it with the -o argument. Thus, the most basic program call would be:

***<p align="center"> ./mbsoptflow -i0 /path/to/my/reference/data/ -i1 /path/to/my/displaced/data/ -o /path/to/dvc/output/</p>***

A full list of available arguments and flags will follow. Right now you need to check the section *extract command line arguments* in *main.cpp* and/or *protocol_parameters.h*. The latter sets the default parameters used that can be overwritten by the arguments.

@Adrian:

Your data a very large and won't fit on a single GPU. It is recommended to use the flag *--mosaic_approximation* to calculate overlapping subvolumes sequentially. That mode assumes that no motions larger than the overlap between subvolumes occur. You will also 

## Exemplary Program Calls

### Push-out tests on bone-screw implant systems (detecting strain and cracks)

*./mbsoptflow -i0 /reference/path/ -i1 /frame1/path/ -o /outpath/ -m /mask/path/ -alpha 0.05 -level 12 -scale 0.8 -norm histogram_linear_mask -prefilter gaussian 1 --median -gpu0 0 --export_warp --conhull_maxz -localglobal 3 -transform1 sqrt --extrapolate --mosaic_approximation -overlap 100 --export_error --bone_quality*

**Lessons:**
<br>
**-gpu0 0** sets the ID (check with *nvidia-smi*) of the first GPU used.
<br>
**-alpha 0.05** controls the relative contribution of the smoothing term. Here, we use as little smoothing as possible and as much smoothing as necessary for locating emerging cracks precisely.
<br>
**-localglobal 3** activates a convolution of the data term. This requires more GPU memory but increases the robustness of the algorithm, especially in poorly textured regions.
<br>
**--mosaic_approximation** activates a sequential approximation of displacements in overlapping subvolumes of image volumes exceeding GPU memory.
<br>
**-norm histogram_linear_mask** sets the method with which to normalize the data to a 0 to 1 interval. Here we stretch 99.9% of the dynamic range of the intensity histogram within the masked ROI. Maintaining brightness constancy is imperative and for zoom tomography and/or low detector counts the assumption may be violated. *linear* provides an additional transformation of Frame1 intensities by projecting extrema in the histogram with a linear transformation onto the corresponding extrema in Frame0.

### Tensile tests on shape memory alloys

**

**Lessons:**
These data only have textures 

### Rearranged sand grains in a flushed capillary

**

In these datasets we can expect rigid body motions exclusively. Thus, smoothing can be substantially increased.

