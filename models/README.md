## 3D U-Net variants and modules.

#### standard 3D U-Net
```python
from unet3d import UNet3D
from layers import DoubleConv

model = UNet3D(in_channels=1, out_channels=1, order='cbl', f_maps=[32,64,128,256], basic_module=DoubleConv, upsample_type='nearest', bias=True)
```
#### Residual module
```python
from layers import ResBlock

model = UNet3D(in_channels=1, out_channels=1, order='cbl', f_maps=[32,64,128,256], basic_module=ResBlock, upsample_type='deconv', bias=True)
```
#### Inverted Residual module(mobilenetv3)
```python
from layers import MBConv

'''InvertedResidual+SE+res: MBConv.__init__(use_fused=False, use_se=True, reduction=4, use_res=True)'''
'''InvertedResidual+res: MBConv.__init__(use_fused=False, use_se=False, reduction=4, use_res=True)'''
model = UNet3D(in_channels=1, out_channels=1, order='cbl', f_maps=[16,32,64,128,256], basic_module=MBConv, upsample_type='deconv', bias=True)
```

#### Fused-MBConve module(efficientNetv2)
```python
from layers import MBConv

'''Fused-MBConve+SE+res: MBConv.__init__(use_fused=True, use_se=True, reduction=4, use_res=True)'''
'''Fused-MBConve+SE: MBConv.__init__(use_fused=True, use_se=True, reduction=4, use_res=False)'''
model = UNet3D(in_channels=1, out_channels=1, order='cbl', f_maps=[16,32,64,128,256], basic_module=MBConv, upsample_type='deconv', bias=True)
```
