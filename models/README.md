# Modules
- <a href="#DoubleConv">DoubleConv</a>
- <a href="#ResBlock">ResBlock</a>
- <a href="#SElayer3D">SElayer3D</a>
## <a id="DoubleConv">DoubleConv</a>
The standard convolutional structure of [3D U-Net](https://arxiv.org/abs/1606.06650).
<img width="90%" height="90%" src="../docs/DoubleConv.png">

## <a id="ResBlock">ResBlock</a>
The residual structure comes from [Superhuman Accuracy on the SNEMI3D Connectomics Challenge](https://arxiv.org/pdf/1706.00120).  
- ResBlock in Encoder
<img width="100%" height="100%" src="../docs/ResBlock_up.png"><br>
- ResBlock in Decoder
<img width="100%" height="100%" src="../docs/ResBlock_down.png">

## <a id="SElayer3D">SElayer3D</a>
The [SE](https://arxiv.org/pdf/1709.01507) module is implemented in 3D.(reference [torchvision SqueezeExcitation](https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py))  
<img width="90%" height="90%" src="../docs/SElayer3D.png">

