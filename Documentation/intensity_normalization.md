### Intensity normalization

Optical flow operates under the assumption of brightness constancy.
Yet, in praxis image intensities may change from scan to scan. This could be due to:
<br>
- low detector counts
- zoom tomography (with larger object motion)
- a chemically or thermically changing sample
- image artefacts
- ...

For quality DVC results emphasis should be put onto assuring brightness constancy as good as possible during preprocessing.
Several normalization options are available via the interface with the ***-norm*** argument followed by:

