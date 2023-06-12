### Alpha parameter

The energy functional that is being minimized consists of a data term and a regularizing smoothness term.
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

Will provide description on how to switch decoupled, isotropic, adaptive,...
