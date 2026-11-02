# Upstream Attribution

This research release uses the official frozen RETFound-Green encoder by Justin
Engelmann and Miguel O. Bernabeu. The encoder is redistributed under the upstream
Custom Non-Commercial Research Licence reproduced in `LICENSE`. The release and
its derived code/weights are supplied for non-commercial research subject to
those terms and applicable upstream terms; this notice does not grant additional
commercial rights.

Engelmann J, Bernabeu MO. Training a high-performance retinal foundation model
with half-the-data and 400 times less compute. Nature Communications 16, 6862
(2025). https://doi.org/10.1038/s41467-025-62123-z

The encoder uses a DINOv2-initialized small vision transformer with register
tokens implemented by timm. Relevant upstream projects and licenses remain
applicable:

- DINOv2: https://github.com/facebookresearch/dinov2
- timm: https://github.com/huggingface/pytorch-image-models
- PyTorch: https://github.com/pytorch/pytorch
- scikit-learn: https://github.com/scikit-learn/scikit-learn

Patient images, patient identifiers, clinical records, and individual prediction
tables are not licensed or distributed by this release. Only aggregate study
information and the requested trained models are included. No clinical fitness
or diagnostic approval is claimed.
