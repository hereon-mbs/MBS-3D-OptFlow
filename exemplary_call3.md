### Rearranged sand grains in a capillary flushed with nanoparticles 

*./mbsoptflow -i0 /reference/denoised/ -i1 /frame1/denoised/ -o /outpath/ -m /localotsu/mask/<br> -alpha 0.2 -norm histogram -scale 0.8 -level 18  -prefilter gaussian 0.5 -gpu0 0<br>--mosaic --export_warp --skip_vectors --binning*

**Lessons:**
<br>
**-m /localotsu/mask/** an externally provided mask limits the data term to the masked in regions. Regions excluded from the mask are interpolated. This is not strictly necessary but in CT data the background texture is often dominated by artefacts which we would like to ignore anyways.
<br>
**-alpha 0.2** In these datasets we can expect rigid body motions exclusively and the grains are quite large. Thus, smoothing can be substantially increased.
<br>
**-level 18** We need to increase the amount of pyramid levels to capture larger rearragements.
<br>
**--mosaic** These data are very large and won't fit on a single GPU. This activates a decomposition of the image volume into smaller subvolumes. In this mode boundaries between the subvolumes are updated after every outer iteration which is substantially slower than optimizing the subvolumes independently with *--mosaic_approximation*. I would try that first although that mode assumes that no motions larger than the overlap between subvolumes occur.
<br>
**--export_warp** creates an additional output of the morphed Frame1 which we are interested in for this study.
<br>
**--skip_vectors** turns off the displacement field output.
<br>
**--binning** skips the last pyramid level and upscales the vectors determined on the previous level. This is sufficient for the task and reduces the computational burden for huge datasets substantially.
<br>
