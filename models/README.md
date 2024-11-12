## 3D U-Net variants and modules.

#### standard 3D U-Net
```python
from unet3d import UNet3D
from layers import DoubleConv

model = UNet3D(in_channels=1, out_channels=1, order='cbl', f_maps=[32,64,128,256], basic_module=DoubleConv, upsample_type='nearest', bias=True)
```
#### residual module
```python
from unet3d import UNet3D
from layers import ResBlock

model = UNet3D(in_channels=1, out_channels=1, order='cbl', f_maps=[32,64,128,256], basic_module=ResBlock, upsample_type='deconv', bias=True)
```
#### mobilenet module
