import torch
import torch.nn as nn

from layers import create_conv_layer, DoubleConv, ResBlock


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 basic_module, pool_type, bias=True, order='cbr'):
        super(Encoder, self).__init__()
        if pool_type == 'max':
            self.pooling = nn.MaxPool3d(kernel_size=2)
        elif pool_type == 'avg':
            self.pooling = nn.AvgPool3d(kernel_size=2)
        elif pool_type == 'none':
            self.pooling = None

        self.basic_module = basic_module(input_dim=in_channels, 
                                         output_dim=out_channels, 
                                         bias=bias, 
                                         order=order)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 basic_module, upsample_type, bias=True, order='cbr'):
        super(Decoder, self).__init__()

        if upsample_type == 'deconv':
            # deconv: default fuse add
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, 
                                               kernel_size=3, stride=2, padding=1, output_padding=1, 
                                               bias=bias)
            self.fuse, in_channels = 'add', out_channels
        elif upsample_type in ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']:
            # Upsample: default fuse concat
            self.upsample = nn.Upsample(scale_factor=2, mode=upsample_type)
            self.fuse = 'concat'
        elif upsample_type == 'none':
            self.upsample = None

        self.basic_module = basic_module(in_channels, out_channels, bias=bias, order=order)

    def forward(self, encoder_features, x):
        if self.upsample is not None:
            x = self.upsample(x)

        if self.fuse == 'concat':
            x = torch.cat((encoder_features, x), dim=1)
        elif self.fuse == 'add':
            x = encoder_features + x

        x = self.basic_module(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, order, f_maps=(64, 128, 256, 512), 
                 basic_module=DoubleConv, pool_type='max', upsample_type='nearest', bias=True) -> None:
        super(UNet3D, self).__init__()

        self.encoders = nn.ModuleList()
        for i, f_map in enumerate(f_maps):
            if i == 0:
                self.encoders.append(Encoder(in_channels, f_map, 
                                             basic_module=basic_module, pool_type='none', 
                                             bias=bias, order=order))
            else:
                self.encoders.append(Encoder(f_maps[i-1], f_map, 
                                             basic_module=basic_module, pool_type=pool_type, 
                                             bias=bias, order=order))
                
        self.decoders = nn.ModuleList()
        reversed_f_maps = list(reversed(f_maps))
        for i, f_map in enumerate(reversed_f_maps):
            if i == 0: continue
            in_f_map = reversed_f_maps[i-1] if upsample_type == 'deconv' else f_map + reversed_f_maps[i-1]
            self.decoders.append(Decoder(in_f_map, f_map,
                                         basic_module=basic_module, upsample_type=upsample_type, 
                                         bias=bias, order=order))
        
        self.final_conv = create_conv_layer(reversed_f_maps[-1], out_channels,
                                            kernel_size=1, stride=1, padding=0, dilation=1, 
                                            bias=True, order='c')

    def forward(self, x):

        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]
        for decoder, encoder_feature in zip(self.decoders, encoders_features):
            x = decoder(encoder_feature, x)

        x = self.final_conv(x)
        return x
    

if __name__=="__main__":

    model = UNet3D(1, 1, order='cbl', f_maps=(16, 32, 64, 128),
                   basic_module=DoubleConv, pool_type='max', upsample_type='nearest', bias=False)
    # model = UNet3D(1, 1, order='cbl', f_maps=(16, 32, 64, 128),
    #                basic_module=ResBlock, pool_type='max', upsample_type='deconv', bias=False)
    
