## Deformation of Ray Cartilage (Basic Functionality)

This demo contains downsampled 8-bit grayscale files of µCT scans from ray cartilage. 
In praxis it is recommended to use higher resolution files and at least 16-bit grayscale information because texture information is crucial to perform high quality DVC.
<br>
<br>
The images in this demo are registered onto a pin (visible near the bottom of the images) that pushes against the cartilage. The demo directory contains the follwing datasets:
- ***Frame01*** serves as a reference for the undeformed cartilage.
- ***Frame02*** is deformed but the cartilage is still intact.
- ***Frame03*** exhibits severe fractures in the material phase.
- A mask on the cartilage for Frame01 (***Mask01***) is incorporated to focus on the material of interest and mask out background artefacts.
- ***raymesh.vtk*** provides a tetrahedral meshing of the mask and serves as a canvas for the projection of results. It can be opened with Paraview.

<br>

### Running the demo

A typical experiment will result in a large quantity of imaged steps. Thus, running the DVC by script is advisable. 
In the root directory of the repository you will find the script ***run_ray_demo.sh*** which upon execution will match ***Frame01*** and ***Frame02***.
Adding ***Frame03*** to the ***EXPERIMENT_LIST*** extends the DVC to an additional time step.
<br>
<br>
The script allows tuning some of the basic DVC parameters:
<br>
- [***LEVEL*** and ***SCALE***](https://github.com/brunsst/MBS-3D-OptFlow/blob/main/Documentation/gaussian_pyramid.md)
- [***ALPHA***](https://github.com/brunsst/MBS-3D-OptFlow/blob/main/Documentation/smoothing_term.md)
