## 3D U-Net variants and modules.

#### standard 3D U-Net
```py
model = UNet3D(in_channels=1, out_channels=1, order='cbl', f_maps=[32,64,128,256], basic_module=DoubleConv, upsample_type='nearest', bias=True)
```
#### residual module

#### mobilenet module
