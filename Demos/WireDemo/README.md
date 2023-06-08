## Tensile Testing of Surface Etched NiTi-Wires

<br>
In this demo we explore how to handle poorly textured samples with gradient based masking and how to minimize bias from larger motion across image boundaries.
The data are also used to give an introduction to the handling of data exceeding GPU memory.
<br>
<br>
The demo directory contains the following data:
<br>
<br>

- **/Frame00/**: an image sequence of a metal wire with etched surface pits
- **/Frame06/**: the same wire with macroscopic tensile strain of 0.00475 applied and registered to the reference frame
- **wire_mesh.vtk**: a tetrahedral mesh of Frame00 with top and bottom cropped by 20 voxels (for minimizing boundary effects)

In the root directory of the repository you will find the script ***run_wire_demo.sh*** which upon execution will match ***Frame01*** and ***Frame06***.
<br>
<br>

### Gradient based masking

Masks are stored in GPU memory combined into a confidence maps with values between zero and one. These values are used to scale the weight of the data term locally. Thus, a binary mask may be converted to a fuzzy mask through blurring.
<br>
<br>
can be exported by adding...
<br>

### Motions over Boundaries

Any motion into or out of the field of view is problematic because the data lack the necessary information to identify the associated displacement vectors. Voxels that are covered by the reference but not by the target frame are by default replaced by the input grayvalue. 
<br>
<br>
As a rule-of-thumb we can expect that the solver will prefer a solution with little motion over large motion when energetically equivalent. This becomes evident with data that have redundant textures like the metal wire in this demo. The interfaces between wire and air are similar enough that the lower cost solution is to indent and unindent the surface instead of stretching the wire across the image boundaries. Bulk parameters that are covered, like the transversal contraction, should remain unaffected. Yet, strain in z-direction is the larger motion (across the image boundaries) and will appear to be close to zero.
<br>
<br>
When additional information on the expected motion is available this can be incorporate in the optimization. In the example at hand we do know from reading out the load cell that the macroscopic strain is 0.00475. The program allows to strain the reference frame in z-direction prior to running the optimization which enables us to look for the smaller transversal strain and local deviations from ideal behavior. This is done by passing the argument:
<div align="center">
  <em><strong>-prestrain_ref 0.00475</em></strong>
</div>
<br>
<br>
The prescribed strain is added back to the result upon completion of the DVC run. The unembedded wire here is expected close to ideal. 


### Combine Local-Global Approach



<br>
