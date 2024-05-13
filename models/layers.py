import torch.nn as nn

__all__ = ['create_conv_layer', 'DoubleConv', 'ResBlock', 'MBConv']


def create_conv_layer(input_dim, output_dim,
                      kernel_size=3, stride=1, padding=0, dilation=1, bias=True,
                      order='cbr'):
    layer = []
    for i, char in enumerate(order):
        if char == 'b':
            if 'c' in order[:i+1]:
                layer.append(nn.BatchNorm3d(output_dim))
            else:
                layer.append(nn.BatchNorm3d(input_dim))
        elif char == 'c':
            layer.append(nn.Conv3d(in_channels=input_dim, out_channels=output_dim,
                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                   dilation=dilation, bias=bias))
        elif char == 'r':
            layer.append(nn.ReLU(inplace=True))
        elif char == 'l':
            layer.append(nn.LeakyReLU(inplace=True))
        elif char == 'e':
            layer.append(nn.ELU(inplace=True))
    return nn.Sequential(*layer)


class DoubleConv(nn.Sequential):
    def __init__(self, input_dim, output_dim, bias, order, 
                 kernel_size=3, stride=1, padding=1, dilation=1):
        super(DoubleConv, self).__init__()
        mid_dim = output_dim // 2 if input_dim < output_dim else output_dim
        # conv1
        self.add_module('conv1', create_conv_layer(input_dim, mid_dim,
                                                   kernel_size, stride, padding, dilation, bias,
                                                   order))
        # conv2
        self.add_module('conv2', create_conv_layer(mid_dim, output_dim,
                                                   kernel_size, stride, padding, dilation, bias,
                                                   order))
class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, bias, order='cbl', 
                 kernel_size=3, stride=1, padding=1, dilation=1):
        super(ResBlock, self).__init__()
        if input_dim != output_dim:
            self.res = create_conv_layer(input_dim, output_dim,
                                         kernel_size=1, stride=1, padding=0, dilation=1,
                                         bias=True, order='c')
        else:
            self.res = None
        self.conv1 = create_conv_layer(output_dim, output_dim,
                                       kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
                                       bias=bias, order=order)
        self.conv2 = create_conv_layer(output_dim, output_dim,
                                       kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
                                       bias=bias, order=order[:-1])
        self.final_activation = create_conv_layer(output_dim, output_dim, order=order[-1])

    def forward(self, x):
        res = self.res(x) if self.res is not None else x
        x = self.conv1(res)
        x = self.conv2(x)
        x += res
        x = self.final_activation(x)
        return x


class SElayer3D(nn.Module):
    def __init__(self, input_dim, reduction=2):
        super(SElayer3D, self).__init__()
        squeeze_channels = input_dim // reduction

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(input_dim, squeeze_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv3d(squeeze_channels, input_dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class MBConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias, order='cbl',
                 expand_ratio=1.25, use_fused=False, use_se=False, reduction=4,
                 use_res=True):
        super(MBConv, self).__init__()
        assert expand_ratio != 1
        expanded_channels = int(output_dim * expand_ratio)

        self.res = create_conv_layer(input_dim, output_dim,
                                     kernel_size=1, stride=1, padding=0, bias=bias, 
                                     order='c') if use_res == True else None

        if use_fused == False:
            # expand
            self.expand = create_conv_layer(input_dim, expanded_channels, 
                                            kernel_size=1, stride=1, padding=0, bias=bias, order=order)
            # depthwise
            self.depthwise = create_conv_layer(expanded_channels, expanded_channels, 
                                            kernel_size=3, stride=1, padding=1, bias=bias, order=order)
        else:
            # use fused
            self.expand = create_conv_layer(input_dim, expanded_channels, 
                                            kernel_size=3, stride=1, padding=1, bias=bias, order=order)
            self.depthwise = None

        self.se =  SElayer3D(expanded_channels, reduction) if use_se == True else None
        # project
        self.project = create_conv_layer(expanded_channels, output_dim, 
                                        kernel_size=1, stride=1, padding=0, bias=bias, order=order[:-1])
        
        self.final_activation = create_conv_layer(output_dim, output_dim, order=order[-1])

    def forward(self, x):
        if self.res is not None:
            res = self.res(x)

        x = self.expand(x)
        if self.depthwise is not None:
            x = self.depthwise(x)

        if self.se is not None:
            x = self.se(x)

        x = self.project(x) + res if self.res is not None else self.project(x)
        x = self.final_activation(x)
        return x
