## Intensity normalization

Optical flow operates under the assumption of brightness constancy.
Yet, in praxis image intensities may change from scan to scan. This could be due to:
<br>
- low detector counts
- zoom tomography (with larger object motion)
- a chemically or thermically changing sample
- image artefacts
- ...

For quality DVC results emphasis should be put onto assuring brightness constancy as good as possible during preprocessing.
Several normalization options are available via the interface with the ***-norm*** argument followed by:

| mode | explanation |
|----|----|
| **none** | Allows to provide an external normalization. Keep to a 0 to 1 range or weird things may happen. |
| **simple** | For 8-bit or 16-bit data; setting 255 or respectively 65535 to 1.0 |
| **histogram** | *(default*) Calculates the cumulative histogram of 99.9% of the dynamic range of Frame 0 and Frame 1 and normalizes to a range of 0 to 1. |
| **histogram_independent** | Calculates separate histograms for Frame 0 and Frame 1. Use this option when intensities changed between scans but the setup is the same. |
| **equalized** | Equalizes the histogram which may help to increase contrast in weakly textured regions. |
| **equalized_independent** | Equalized with independent histograms. |
| **equalized_mask** | Performs equalization exclusively in the region provided with *-m*. E.g., this option was used for tendon matching because neighboring bone that was of no interest would dominate the contrast in the images. |
| **histogram_linear** |  Applies a linear least square fits to maxima and minima of both histograms to match them. This option was used for **SynchroLoad**. With the low counts in the projections histograms were often stretched/compressed. An additional textfile output is generated to judge the histogram correlation. |
| **histogram_linear** | Same as above with equalization. |

<br>

#### Additional information
With the ***-transform1*** argument followed by a string an additional arithmetic operation may be performed before normalization
(available: *sqrt*, *log*, *minus_log*, *exp*, *pow2*, *pow3*,...).
This may be relevant if image equalization equivalent to Fiji/ImageJ is desired. ImageJ performs histogram equalization on the square root of intensities.
<br>
Arithmetic operations after image normalization are available via the ***-transform2*** argument.

#### Compile time parameters

| parameter | type | default | explanation |
|-----------|------|---------|-------------|
| extrapolate_intensities | bool | false | Not advised. When true, intensities outside the dynamic range are scaled to values smaller/larger than 0/1. This may yield unexpected behavior (and create large data term values for outlier voxels). Option is not available with histogram equalization. | 
| ignore_zero | bool | true | By default voxels with intensity 0 are ignored in calculating the stack histogram because these may be values outside the reconstructed region of interest. |
| rescale_zero | bool | false | If true, intensities that are 0 before normalization will remain 0. |
| sqrt_equalization | bool | false | Option to perform less extreme, ImageJ-like histogram equalization. Switchable with *--sqrt_equalize*. Potentially discontinued. Equalization is rarely the smartest option anyways. |
| smoothed_equalization | int | 0 | smooth the equalization with a mean filter when != 0 |

