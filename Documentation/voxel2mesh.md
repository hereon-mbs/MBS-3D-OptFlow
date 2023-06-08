## Voxel2Mesh

The program provides a simplified way of mapping displacement vectors onto *MEDIT* or *VTK* meshes.
By default a minimization of rigid body motion is performed over all surface tetrahedra, i.e., displacement vectors are transformed into deformation vectors.

<br>

Mandatory arguments/inputs:

| argument | value | explanation |
|---|---|---|
| **-i_mesh** | string | Path to a tetrahedral mesh in **VTK** format or **MEDIT** format. |
| **-i_disp** | string | Path to a directory contain subfolder *dx*,*dy* and *dz* with displacement vector components as tif image sequence. |

<br>

Optional arguments:

| argument | value | explanation |
|---|---|---|
| **-i_mesh** | string | Path to a tetrahedral mesh in **VTK** format or **MEDIT** format. |
| **-i_disp** | string | Path to a directory contain subfolder *dx*,*dy* and *dz* with displacement vector components as tif image sequence. |
