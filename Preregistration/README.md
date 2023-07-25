# WBBRegistration_v0.2

This code offers prototype functionality to perform GPU supported rigid body registration of 3D image stacks before running the DVC algorithm.
A simple line search is performed to maximize the correlation between the two frames. 
Line search may be followed by an additional gradient ascent optimization.

Typically, I maximize correlation between adjacent frames and then apply the fit cumulatively. 
This needs to be followed by a crop to the biggest common field of view for which no functionality is provided here.

The program can be build separately from the *build_wbbregistration.sh* script and can be run from the terminal.

<br>

**Warning! When rotations are applied without any translation bugs may occur. This can be prevent by adding a subvoxel translation to the reference.**

<br>

|argument|parameter|explanation|
|--------|---------|-----------|
| -i0    | string  | *mandatory* path to tif image sequence with reference |
| -i1    | string  | *mandatory* path to tif image sequence with displaced data |
| -m     | string  | *optional* path to regions considered during registration |
|        |         |        |
| -range | 2x int  | *otional* select a subset of slices used for registration (for acceleration or to focus on the center) |
| -zoffset | int   | *optional* shifts frame1 to read in a comparable range |
| -guess | 6x float | *optional* provide an initial guess or the previous solution as [delta_x,delta_y,delta_z,jaw,roll,pitch] |
| -gpu0  | int | *optional* select the device to be used |

<br>

By default only translation is optimized. Rotations need to be activated with the corresponding flag:

<br>

|flag|explanation|
|--------|-----------|
| --jaw  | activates optimization of first rotational degree of freedom. |
| --roll  | activates optimization of second rotational degree of freedom. |
| --pitch  | activates optimization of third rotational degree of freedom. |
| --samples | creates projections and slices of the registration result for screening |
| --optimize | without the flag only line search is performed. With the flag line search is followed by a gradient ascent. |
| --transform_only | apply the guess without further optimization |

<br>

A typical workflow would be to start optimizing translation and rotation with some central slices:

*./wbbregistration -i0 /path/frame0/ -i1 /path/frame1/ -guess -8.538 9.08 -88.9 -0.237 -5.084 -3.652 --samples -gpu0 3 -range 200 1400 --optimize --jaw --roll --pitch*

... and when satisified summing up and applying the result by running:

*./wbbregistration -i0 /path/frame0/ -i1 /path/frame1/ -guess -8.538 9.08 -88.9 -0.237 -5.084 -3.652 --samples --transform_only*


