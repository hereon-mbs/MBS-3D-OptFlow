

### Alpha parameter

The energy functional that is being minimized consists of a data term and a regularizing smoothness term of the form:

$$ E_{smooth}(u,v)= \int_\Omega \Psi (\left\lvert \nabla_3u \right\rvert^2 + \left\lvert \nabla_3v \right\rvert^2) dx $$

During preprocessing 99.9% of the dynamic range are normalized to a 0 to 1 intensity range.
This allows expressing the contribution of the smoothing term in relative terms through the alpha parameter.
By default moderate smoothing at 7% is performed:
<br>
<br>
| argument | type | default | explanation |
|----|----|----|----|
| **-alpha** | float | 0.07 | Relative weight of the smoothing term. Note that the data term operates on the intensity gradient, i.e., low contrast regions appear to be smoothed out more than high contrast regions. Consider histogram equalization or other contrast enhancements when your feature of interest lies in a low contrast image region or other features, like a metallic object, dominate the contrast in your images. |

<br>

### Available smoothness terms

The default solver and thus best maintained solver is [optflow_gpu3d.cu](../Source/Solver/optflow_gpu3d.cu). Smoothness terms for this solver are implemented by [smoothnessterm_gpu3d.cuh](../Source/Derivatives/smoothnessterm_gpu3d.cuh) with various discrete [derivatives](derivatives.md) available. As discontinuities are expected at object boundaries piecewise smooth regularizers are best suited and all implementations are flow-driven following a total variational scheme ([modified L1 minimization](https://www.mia.uni-saarland.de/Publications/brox-eccv04-of.pdf)), i.e., the implemented smoothness term is of the form:

$$ \Psi(s^2)=\sqrt{s^2+\epsilon^2} $$

and thus the returned spatial derivative of functions in smoothnessterm_gpu3d.cuh being of the form:

$$ {1 \over 2\sqrt{(s+\epsilon)}} $$

The recommended default regularizer used is an [anisotropic flow-driven regularizer](https://doi.org/10.1023/A:1013614317973). The design can be changed at runtime with the following flags:

| flag | explanation |
|------|-------------|
| **--isotropic** | Changing to an isotropic smoothness term will allow for more flow across image edges. |
| **--decoupled** | The decoupled version of each smoothness term implementation returns an independent value for each spatial dimension. |
| **--adaptive** | With an anisotropic smoothness term the preferential smoothing direction is defined via the principal image axis. In adaptive mode edge orientation may be considered which may enhance stepwise changes at object boundaries further. The costly and usually only a minor improvement to the default case. The working implementation identifies edges from a blurred gradient image. Identification from the structure tensor is still buggy*. |
| **--complementary** | Complementary version of the smoothness term following [Zimmer et al 2011](https://doi.org/10.1007/s11263-011-0422-6). |


\* *My current guess would be that there is no convolution before identification of the Eigenvalues, which will only yield the maximal Eigenvalue.*
