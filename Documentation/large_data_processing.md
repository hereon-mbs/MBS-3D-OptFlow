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

### Passable Arguments

| argument | value | default | explanation |
|----------|-------|---------|-------------|
|

### Advanced Parameters
