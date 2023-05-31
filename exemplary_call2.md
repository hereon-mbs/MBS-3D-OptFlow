### Tensile tests on wires made of shape memory alloys

*./mbsoptflow -i0 /frame0/ -i1 /frame1/ -o /outpath/ -m none -gradientmask 0 0.05 <br>-alpha 0.05 -norm histogram_independent -level 10 -scale 0.8 --skip_masking --export_warp -prefilter gaussian 0.5 -localglobal 3 -gpu0 2 -prestrain_ref 0.001 0.5*

**Lessons:**
<br>
**-m none -gradientmask 0 0.05** These data only have textures on the interface of the wire where etched pits can be found. Instead of using an external mask we use the internal option to calculate the gradient magnitude. The first number provided with the *-gradientmask* argument defines the sigma value for Gaussian blurring of the gradient image, whereas the second value defines the percentile of voxels used. In this case we use the 5% of voxels with the largest gradient magnitude.
<br>
**--skip_masking** This should be default by now. In the first version masked out values were assumed to be stationary by default in the displacement vector field. That turned out to be only useful for visualization purposes.
<br>
**-prestrain_ref 0.001 0.5** When performing a tensile test the sample extends beyond the field of view at the top and bottom of the image stack. Even more, the motion across this boundary is larger than the lateral contraction. We can aid the algorithm in finding a reasonable solution by straining the reference image by the expected tensile strain during the preprocessing step. This also maximizes the amount of information used from Frame1. The result is corrected for the prescribed strain. Here, we strain the reference image by 0.1% around the image center (*0.5*). 
<br>
**-norm histogram_independent** Instead of using a cumulative histogram from Frame0 and Frame1 for normalization adding *_independent* normalizes Frame0 and Frame1 using their individual histograms.

<br>
