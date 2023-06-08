## Voxel2Mesh

The program provides a simplified way of mapping displacement vectors onto *MEDIT* or *VTK* meshes.
By default a convolution with a Gaussian kernel (sigma=3.0) is applied to displacement field. 
Subsequently, a minimization of rigid body motion is performed over all surface tetrahedra, i.e., displacement vectors are transformed into deformation vectors.


### Mandatory inputs

| argument | value | explanation |
|---|---|---|
| **-i_mesh** | string | Path to a tetrahedral mesh in **VTK** format or **MEDIT** format. |
| **-i_disp** | string | Path to a directory contain subfolder ***dx***, ***dy*** and ***dz*** with displacement vector components as tif image sequence. |

<br>

### Available mappings

Mappings are derivatives calculated from the passed vector field. They are invoked without preceding dash.
<br>
<br>
Default derivatives are calculate as optimal 5-tap kernel derivatives according to [Farid and Simoncelli](https://www.cns.nyu.edu/pub/lcv/farid03-reprint.pdf).
This uses 3D information more efficiently than conventional finite difference stencils.

| argument | explanation |
|---|---|
| Exx | Exx component of the [Green-Lagrange strain tensor](https://www.continuummechanics.org/greenstrain.html) |
| Eyy | respective Eyy component |
| Ezz | respective Ezz component |
| Exy | respective Exy component |
| Eyz | respective Eyz component |
| Exz | respective Exz component |
| volstrain | Volumetric strain calculated from the deformation gradient *F* as det(F)-1 with optimal 5-tap kernel derivatives. |
| maxshear | Maximum shear strain calculated from Green-Lagrange strain tensor using optimal 5-tap kernel for derivative calculations. |
| max_pstrain | [3D maximum principal strain](https://www.continuummechanics.org/principalstrain.html) calculated from the invariants of the Green-Lagrange strain tensor.|

<br>

### Optional arguments

| argument | value | explanation |
|---|---|---|
| **-sigma** | float | Change the sigma value for the Gaussian convolution. 0.0 deactivates the filter. |
| **-vxl** | float | Used for transforming from voxel scale to physical scale. Mutliplies the vertex coordinates and displacement vector components with the voxel size provided with this paramter. |

<br>

### Optional flags

| &nbsp; &nbsp; flag &nbsp; &nbsp; | explanation |
|---|---|
| **--taubin** | Applies ***10 iterations*** of Taubin smoothing to all surface triangles with ***lambda=0.5*** and ***mu=-0.53***. |
| **--vertices**  &nbsp; &nbsp; &nbsp; &nbsp; | By default displacements are interpolated for the center of gravity of each tetrahedra cell. The flag allows interpolating on vertex coordinates instead. This is required to visualize the vectors as glyphs with Paraview |
