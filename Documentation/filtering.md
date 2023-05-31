### Prefilter

Images need to be continuously derivable which can be assured by Gaussian filtering as a preprocessing step. 
If input images were properly denoised, sigma of the Gaussian kernel can often be reduced to 0.5. 
This is preferable because blurring objects increases the spatial uncertainty in the calculated displacements.
The following options are available:


| argument | value | explanation |
|--------|------------------|-----------|
| **-iter**|integer|(*optional*, default=4) number of denoising iterations|
