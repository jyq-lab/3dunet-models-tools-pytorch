## Network
- [3D U-Net](#3d-u-net)

### 3D U-Net

## Modules
- [DoubleConv](#doubleconv)
- [ResBlock](#resblock)
- [SElayer3D](#selayer3d)
- [MBConv](#mbconvfused-mbconv)

### DoubleConv
The standard convolutional structure of [3D U-Net](https://arxiv.org/abs/1606.06650).
<img width="90%" height="90%" src="../docs/DoubleConv.png">

### ResBlock
The residual structure comes from [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120).  
- ResBlock in Encoder
<img width="100%" height="100%" src="../docs/ResBlock_up.png"><br>
- ResBlock in Decoder
<img width="100%" height="100%" src="../docs/ResBlock_down.png">

### SElayer3D
The [SE](https://arxiv.org/pdf/1709.01507) module is implemented in 3D.(reference :fire:[torchvision SqueezeExcitation](https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py))  
<img width="90%" height="90%" src="../docs/SElayer3D.png">

### MBConv(Fused-MBConv)
[EfficientNet v2](https://arxiv.org/pdf/2104.00298)
