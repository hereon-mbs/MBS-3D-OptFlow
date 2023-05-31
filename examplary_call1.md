### Push-out tests on bone-screw implant systems (detecting strain and cracks)

*./mbsoptflow -i0 /reference/path/ -i1 /frame1/path/ -o /outpath/ -m /mask/path/<br>-alpha 0.05 -level 12 -scale 0.8 -norm histogram_linear_mask -prefilter gaussian 1 --median -gpu0 0<br>--export_warp --conhull_maxz -localglobal 3 -transform1 sqrt --extrapolate --mosaic_approximation -overlap 100 --export_error --bone_quality*

**Lessons:**
<br>
**-gpu0 0** sets the ID of the first GPU used. You can check the IDs of your GPUs with *nvidia-smi*.
<br>
**-alpha 0.05** controls the relative contribution of the smoothing term. Here, we use as little smoothing as possible and as much smoothing as necessary for locating emerging cracks precisely.
<br>
**-localglobal 3** activates a convolution of the data term. This requires more GPU memory but increases the robustness of the algorithm, especially in poorly textured regions. The integer defines the radius of the convolution kernel. Near optimal kernels are available with a radius of up to 4.
<br>
**--mosaic_approximation** activates a sequential approximation of displacements in overlapping subvolumes of image volumes exceeding GPU memory.
<br>
**-norm histogram_linear_mask** sets the method with which to normalize the data to a 0 to 1 interval. Here we stretch 99.9% of the dynamic range of the intensity histogram within the masked ROI. Maintaining brightness constancy is imperative and for zoom tomography and/or low detector counts the assumption may be violated. *linear* provides an additional transformation of Frame1 intensities by projecting extrema in the histogram with a linear transformation onto the corresponding extrema in Frame0.

<br>
