We provide pretrained models that can be loaded via
```py
from ldctbench.hub import load_model
net = load_model("<method>")
```
where `<method>` is one of the method names provided in [this table][implemented-algorithms] (e.g., `cnn10`, `redcnn`, `wganvgg`, ...) or a member of `ldctbench.hub.Methods` (e.g., `ldctbench.hub.Methods.CNN10`). To apply it, make sure that CT images are stored with an offset of 1024, i.e. air has a value of ~24.

```python
import numpy as np
from ldctbench.hub import load_model, Methods
from ldctbench.evaluate import preprocess, denormalize

method = Methods.RESNET # method="resnet" also works
# Setup model
net = load_model(method)
# Define image
x = # ... some numpy array of shape [1, 512, 512] you wish to denoise
# Preprocess and normalize input
x_t = preprocess(x, method=method)
# Apply network
y_hat = net(x_t)
# Denormalize
y_hat = denormalize(y_hat, method=method)
```

We provide a comprehensive example on how to denoise DICOM data using the pretrained models in [this example](examples/denoise_dicoms.md).
