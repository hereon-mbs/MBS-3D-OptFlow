## Large Data Processing

Processing of data exceeding available GPU memory is managed by protocols in the header file [*mosaic_approximation.h*](../Source/Protocols/mosaic_approximation.h).
Two functions are currently interfaced for evaluating a mosaic of overlapping patches:

***run_sequential_mosaic***:
- Does not communicate between patches. Patches have a Dirichlet boundary to previously solved patches.
- Should work if there are only small objects and is reasonable fast.
- Direction alternates by level.

<br>

***run_singleGPU_mosaic*** (*not maintained actively*):
- Updates every patch within an outer iteration. 
- Dirichlet boundaries maintain the previous result. (better fit but bigger deviation from default and slow.)

<br>

### Switchable Flags

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | explanation |
|----------|-------------|
| **--mosaic_approximation** | (*recommended*) activates sequential mosaic processing by switching parameters *mosaic_decomposition* and *sequential_approximation* to *true* |
| **--mosaic** | Switches mosaic_decomposition to *true* which activates the singleGPU_mosaic. |

### Interfaced Arguments

| argument | value | default | explanation |
|----------|-------|------|-------|
| **-overlap** | int | 100 | Allows changing the extent of the overlap between patches. This may be tuned for efficiency according to size of objects and speed of motion between frames. |
| **-cut_dim** | int | -1 | Deactivated with -1. Allows defining a dimension (0,1 or 2) from which patches are preferably cut. You may want to use this option when you have less motion in one dimension than the others where you want to avoid overlap regions. |

### Compilable Parameters

| argument | value | default | explanation |
|----------|-------|------|-------|
| **max_nstack** | int | -1 | Maximal allowed amount of voxels in a patch. The default of -1 activates auto estimation from available GPU memory. The recommendation is to have at least 2*(max_shift+iter_sor+1).|
| **memory_buffer** | int | 256 | MB kept free on GPU as backup when auto estimating a patch size that can be fit into memory. You may want to reduce this value and recompile when running the code on a home computer. |


### Miscellaneous

| &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;flag&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | explanation |
|----------|-------------|
| **--reorder** | Transpose the axis to have the mosaic cuts in the z-dimension. This accelerates copying between patches (cascading layout) which is unnecessary for approximation mode. |

| argument | &nbsp;&nbsp;value&nbsp;&nbsp; | explanation |
|----------|-------|-------|
| **-mosaic** | 2x int | Activates mosaic decomposition allowing to define the maximum patch size with the first integer value and the overlap in voxels with the second integer. |

