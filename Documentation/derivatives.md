### Available derivatives

Discretized temporal derivative components used throughout the code are calculated with a simple forward difference scheme using Frame 0 and Frame 1. Spatial derivative components are by default calculated with a near optimal 5-tap kernel according to [Farid and Simoncelli](). Incorporating off-axis components into derivative calculations is beneficial because it increases the robustness of the solution against noise and outliers.
<br>

Alternative derivatives are available that can be set for the smoothness term with the ***-smoothness_stencil*** argument followed by a string defining the derivative type.
<br>

The following discretization schemes are currently available:

|keyword|explanation|
|-------|-----------|
| centraldifference | *self-explanatory* |
| forwarddifference | *self-explanatory* |
| Barron | Implements a fourth-order finite difference scheme. |
| Farid3 | Faster 3-tap kernel instead of 5-tap kernel. |
| Farid5 | *Default* |
| Farid7 | Higher accuracy 7-tap kernel instead of 5-tap kernel. |
| Farid9 | Higher accuracy 9-tap kernel instead of 5-tap kernel. |

<br>

The argument ***-spatiotemporal_stencil*** allows changing the derivative used in data term calculations.
<br>

The following discretization schemes are currently available:

|keyword|explanation|
|-------|-----------|
| HornSchunck | (*legacy mode*) Spatial and temporal derivatives follow the original layout of [Horn and Schunck](https://en.wikipedia.org/wiki/Horn%E2%80%93Schunck_method). |
| Ershov | Spatial and temporal derivatives follow the 2D implementation of [A. Ershov](https://github.com/axruff/cuda-flow2d), i.e., a central difference scheme in the spatial dimensions and a forward difference scheme in the temporal domain. |
| Barron | Change to a 4th order finite difference scheme for the spatial dimensions. |
| Farid3 | Faster 3-tap kernel instead of 5-tap kernel. |
| Farid5 | *Default* |
| Farid7 | Higher accuracy 7-tap kernel instead of 5-tap kernel. |
| Farid9 | Higher accuracy 9-tap kernel instead of 5-tap kernel. |
