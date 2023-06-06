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
- **wire_mesh.vtk**: a tetrahedral mesh of Frame00 with top and bottom cropped by 10 voxels (for minimizing boundary effects)

In the root directory of the repository you will find the script ***run_wire_demo.sh*** which upon execution will match ***Frame01*** and ***Frame06***.
<br>
<br>
The script allows tuning the following parameters:
