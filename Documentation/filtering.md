### Prefilter

Images need to be continuously derivable which can be assured by Gaussian filtering as a preprocessing step. 
If input images were properly denoised, sigma of the Gaussian kernel can often be reduced to 0.5. 
This is preferable because blurring objects increases the spatial uncertainty in the calculated displacements.
The following options are available:

| argument &nbsp; | &nbsp; &nbsp; default &nbsp; &nbsp; &nbsp; | &nbsp; explanation &nbsp; |
|---|---|---|
| **-prefilter** | gaussian 1.0 | Follow the argument either with *none* or a string followed by a float defining the kernel. Available: *gaussian*, *median*, *median_simple* |
| **-prefilter2** | none | Activates a prefilter that is only applied to the zero-level of the image pyramid. |

