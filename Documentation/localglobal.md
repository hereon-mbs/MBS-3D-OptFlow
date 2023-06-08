## Combined Local-Global Optical Flow

Variational methods minimize a global energy and hence belong to the family of global DVC algorithms. Global approaches to DVC achieve kinematic and mechanic compatibility of the solution by an intrinsic spatial coupling. Solving for the entire domain at once offers better accuracy than a local approach with correlation windows but comes at the cost of local precision. The combined local-global approach by **[Bruhn, Weickert and Schnörr](https://www.mia.uni-saarland.de/Publications/bruhn-ijcv05c.pdf)** attempts to increase the robustness of optical flow based approaches versus noise. In essence, they describe a convolution of the data term introducing a correlation length. This does not only increase the robustness against noise but is also beneficial when missing textures in the data. 


### Implementation at runtime

The combined local-global method is activated at runtime with the following arguments:

<br>

| &nbsp; argument &nbsp; | value | explanation |
|-----------|---|---|
| **-localglobal**  &nbsp; | int  &nbsp; | Improved version using near-optimal interpolation kernels by **[Farid and Simoncelli](https://www.cns.nyu.edu/pub/lcv/farid03-reprint.pdf)**. Follow the argument with an integer value defining the radius of the interpolation kernel. The mode is limited to kernels of radius 1,2,3 or 4. |
| **-localglobal_gauss** &nbsp; | float  &nbsp; | Conventional implementation of the combined local-global approach without upper limit. Follow the argument with a value defining the sigma value for the Gaussian convolution kernel. |

<br>

### Background parameters

The algorithm offers various ways for implementing a convolution during compile time in the *protocol_parameters.h*:

<br>

| parameter &nbsp; | &nbsp; &nbsp; default &nbsp; &nbsp; &nbsp; | &nbsp; explanation &nbsp; |
|---|---|---|
| **localglobal_dataterm** | false | Activates combined local-global calculations, i.e., additional memory is required because the dataterm cannot be calculated on the fly. |
| **localglobal_mode** | "Farid" | Defines the layout of the convolution kernel. Available are the original implementation via Gaussian convolution (*"Gauss"*) and an improved version using near-optimal interpolation kernels by **[Farid and Simoncelli](https://www.cns.nyu.edu/pub/lcv/farid03-reprint.pdf)** (*"Farid"*) |
| **localglobal_fading_sigma** | {} | ***Discontinued*** Vector of sigma values that can be used to fade out the coupling when approaching the solution. |
| **localglobal_sigma_data** | 1 | Sigma of the Gaussian convolution kernel with *localglobal_mode="Gauss"* or radius of the interpolation kernel with *localglobal_mode="Farid"*. Available are 3-tap to 9-tap kernels (*localglobal_sigma_data=1,2,3 or 4*) |
