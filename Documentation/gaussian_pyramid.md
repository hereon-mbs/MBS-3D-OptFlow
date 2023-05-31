## Parameters for controling the image pyramid

The algorithm operates on a [Gaussian image pyramid](https://en.wikipedia.org/wiki/Pyramid_(image_processing)). 
This is necessary for identifying larger displacements because convex optimization is only strictly valid for displacements <1 voxel.
The following arguments allow tuning the underlying image pyramid:

| argument | type | default | explanation |
|----|----|----|----|
| **-level** | int | 10 | Sets the amount of pyramid levels.|
|**-scale**| float | 0.9 | Defines the relative amount of downsampling between pyramid levels and thereby the slope of the image pyramid. |

<br>

### Expert parameters

Parameters for the expert user that are tunable during compile time:

| parameter | type | default | explanation |
|----|----|----|----|
| min_edge | int | 4 | Minimal edge length of a downsampled volume image. Will stop the downsampling in that dimension. |
| alpha_scaling | bool | false | Undocumented option switchable during runtime with the *--scaled_alpha flag* (scales alpha for each pyramid level)  |
| interpolation_mode | string | cubic_unfiltered | Undocumented. Available: *linear*, *cubic*, *linear_filtered*, *cubic_filtered*, *linear_unfiltered*, *linear_antialiasing*, *cubic_antialiasing* |
| scaling_mode | string | custom | Undocumented. Available: *custom*, *Ershov*, *Liu* |

During runtime the argument *-pyramid* followed by a string for the scaling mode and a string for the interpolation mode allows tuning the fundamental layout of the image pyramid. This is usually not necessary.
