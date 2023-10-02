import torch
import torch.nn.functional as F
import torchvision.models.resnet as resnet

class BasicBlockWithDilation(torch.nn.Module):
    """
    Redidual block. 
    Workaround for prohibited dilation in BasicBlock in 0.4.0.
    The original ResNet does not allow a dilated convolution so that the TA
    built his own class for the residual block with an option for dilated convolution. 
    This function was copied from the PyTorch documentation. The only change is the
    dilation option. 
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockWithDilation, self).__init__()
        
        # Create default 2d batchnorm layer. 
        if norm_layer is None:
            norm_layer = torch.nn.BatchNorm2d
        # Check input. 
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        # Layer implementations. 
        self.conv1 = resnet.conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = torch.nn.ReLU()
        # The following line has an add. dilation option, which is not implemented 
        # in the basic block of the original ResNet according to the PyTorch doc.  
        self.conv2 = resnet.conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """ Implementation of the residual block. """
        identity = x
        
        # Residual block. 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # Check if the feature map resolution changed. 
        if self.downsample is not None:
            identity = self.downsample(x)
        # Add residual block input to worked on output. 
        out += identity
        # Apply relu on the combined feature map. 
        out = self.relu(out)
        
        return out

_basic_block_layers = {
    'resnet18': (2, 2, 2, 2),
    'resnet34': (3, 4, 6, 3),
}

def get_encoder_channel_counts(encoder_name):
    """
    Returns the number of encoder output channels (bottleneck layer) and 
    the number of channels after the 2nd convolution (channel: 56x56x64).
    The feature map after the 2nd convolution is used as input for the 
    decoder module. 
    """
    # Check ResNet mode used. 
    is_basic_block = encoder_name in _basic_block_layers
    # No. of encoder output channels. 
    ch_out_encoder_bottleneck = 512 if is_basic_block else 2048
    # No. of channels after the 2nd conv. layer. 
    ch_out_encoder_4x = 64 if is_basic_block else 256

    return ch_out_encoder_bottleneck, ch_out_encoder_4x

class Encoder(torch.nn.Module):
    def __init__(self, name, **encoder_kwargs):
        super().__init__()
        encoder = self._create(name, **encoder_kwargs)
        # Remove the average pooling and fc layer, these are only needed for classification. 
        del encoder.avgpool
        del encoder.fc
        self.encoder = encoder

    def _create(self, name, **encoder_kwargs):
        if name not in _basic_block_layers.keys():
            # Specifies the resnet model, same as resnet.name
            # Returns the attribute, which is the specific resnet model in this case. 
            fn_name = getattr(resnet, name)
            model = fn_name(**encoder_kwargs)
        else:
            # special case due to prohibited dilation in the original BasicBlock
            # Pop kwarg entry and return it if existant, return default (2nd argument) otherwise 
            pretrained = encoder_kwargs.pop('pretrained', False)
            progress = encoder_kwargs.pop('progress', True)
            # Design resnet model. 
            model = resnet._resnet(
                name, BasicBlockWithDilation, _basic_block_layers[name], pretrained, progress, **encoder_kwargs
            )
        
        # Replace stride with a dilated convolution. 
        replace_stride_with_dilation = encoder_kwargs.get('replace_stride_with_dilation', (False, False, False))
        assert len(replace_stride_with_dilation) == 3
        # Check different layers for replacement. 
        # Tuples define the values of padding and dilation (height, width)
        # Using a single int instead of a 2 tuple uses the value for both height and width. 
        if replace_stride_with_dilation[0]:
            # Acces the first weight layer (a conv. layer) of the second residual layer. 
            model.layer2[0].conv2.padding = (2, 2)
            model.layer2[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[1]:
            model.layer3[0].conv2.padding = (2, 2)
            model.layer3[0].conv2.dilation = (2, 2)
        if replace_stride_with_dilation[2]:
            model.layer4[0].conv2.padding = (2, 2)
            model.layer4[0].conv2.dilation = (2, 2)
        
        return model

    def update_skip_dict(self, skips, x, sz_in):
        # Use division by the original image height to compute the 
        # current ResNet layer index. ResNet feature maps decrease by a 
        # factor of 2 with each residual layer. 
        rem, scale = sz_in % x.shape[3], sz_in // x.shape[3]
        # Check if size of residual layer did decrease feature map size by 2x. 
        # This is required by the ResNet design. 
        assert rem == 0
        # Add feature map to feature pyramid. 
        skips[scale] = x

    def forward(self, x):
        """
        DeepLabV3+ style encoder
        :param x: RGB input of reference scale (1x)
        :return: dict(int->Tensor) feature pyramid mapping downscale factor to a tensor of features
        """
        # Store the feature pyramid after each feature map decrease (after each ResNet layer).
        out = {1: x} 
        # Width of input image.
        sz_in = x.shape[3] 

        # First conv. (image conv.)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        # Store first feature map.
        self.update_skip_dict(out, x, sz_in) 
        x = self.encoder.maxpool(x)
        # Store second feature map. Feature map is max. pooled before it is 
        # fed to the residual layers. 
        self.update_skip_dict(out, x, sz_in)

        # Residual layers. 

        x = self.encoder.layer1(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer2(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer3(x)
        self.update_skip_dict(out, x, sz_in)

        x = self.encoder.layer4(x)
        self.update_skip_dict(out, x, sz_in)

        return out

# class DecoderDeeplabV3p(torch.nn.Module):
#     def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
#         super(DecoderDeeplabV3p, self).__init__()

#         # TODO: Implement a proper decoder with skip connections instead of the following
#         self.features_to_predictions = torch.nn.Conv2d(bottleneck_ch, num_out_ch, kernel_size=1, stride=1)

#     def forward(self, features_bottleneck, features_skip_4x):
#         """
#         DeepLabV3+ style decoder
#         :param features_bottleneck: bottleneck features of scale > 4
#         :param features_skip_4x: features of encoder of scale == 4
#         :return: features with 256 channels and the final tensor of predictions
#         """
#         # TODO: Implement a proper decoder with skip connections instead of the following; keep returned
#         #       tensors in the same order and of the same shape.
#         features_4x = F.interpolate(features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False)
#         predictions_4x = self.features_to_predictions(features_4x)
#         return predictions_4x, features_4x

""" Decoder for task 1 and 2. """

class DecoderDeeplabV3p(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3p, self).__init__()
        # DCNN 1x1 convolution (only necessary if the no. of channels should be increased from 64 to 256). 
        # self.dcnn_conv = torch.nn.Conv2d(64, 256, kernel_size=1, stride=1)
        # 3x3 on concatenated feature maps. 
        self.conv_3x3 = torch.nn.Conv2d(bottleneck_ch+skip_4x_ch, num_out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels and the final tensor of predictions
        """
        # Upsample ASPP output 4x. 
        features_4x = F.interpolate(features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False)
        # DCNN 1x1 convolution (only necessary if the no. of channels should be increased from 64 to 256). 
        # features_dcnn = self.dcnn_conv(features_skip_4x)
        # Tensor concatination. 
        concat_tensor = torch.cat((features_4x, features_skip_4x), 1)
        # 3x3 Convolution
        predictions_4x = self.conv_3x3(concat_tensor)
        return predictions_4x, features_4x

""" ASPP module and its submodules. """

class ASPPpart(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

class ImagePooling(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels),
        )

class ASPP(torch.nn.Module):

    def __init__(self, in_channels, out_channels, rates=(3, 6, 9)):
        super().__init__()
        # 1x1 convolution. 
        self.conv1 = ASPPpart(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        # Dilated convolutions. 
        self.conv_r1 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv_r2 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv_r3 = ASPPpart(in_channels, out_channels, kernel_size=3, stride=1, padding=9, dilation=9)
        # Image pooling. 
        self.im_pool = ImagePooling(in_channels=in_channels, out_channels=out_channels)
        # 1x1 convolution before hand-over to the decoder (is fed the concatenated tensor from the ASPP module). 
        self.conv_out = ASPPpart(out_channels*5, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        
    def forward(self, x):
        # 1x1 convolution. 
        c1 = self.conv1(x)
        # Dilated convolutions. 
        cr1 = self.conv_r1(x)
        cr2 = self.conv_r2(x)
        cr3 = self.conv_r3(x)
        # Image pooling. 
        h = x.shape[2]
        w = x.shape[3]
        ip = F.interpolate(self.im_pool(x), size=(h, w), mode='bilinear', align_corners=False)
        # Concatenate individual tensors. 
        concat_tensor = torch.cat((c1, cr1, cr2, cr3, ip), 1)
        # Convolve all ASPP tensors. 
        out = self.conv_out(concat_tensor)
        return out

""" Task 3. First decoder of the SA module. """

class DecoderDeeplabV3pSA(torch.nn.Module):
    def __init__(self, bottleneck_ch, skip_4x_ch, num_out_ch):
        super(DecoderDeeplabV3pSA, self).__init__()
        # DCNN 1x1 convolution (only necessary if the no. of channels should be increased from 64 to 256). 
        # self.dcnn_conv = torch.nn.Conv2d(64, 256, kernel_size=1, stride=1)
        # 3x3 on concatenated feature maps. 
        self.conv_3x3 = torch.nn.Conv2d(bottleneck_ch+skip_4x_ch, num_out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, features_bottleneck, features_skip_4x):
        """
        DeepLabV3+ style decoder
        :param features_bottleneck: bottleneck features of scale > 4
        :param features_skip_4x: features of encoder of scale == 4
        :return: features with 256 channels, output for the SA module (concat tensor)
        """
        # Upsample ASPP output 4x. 
        features_4x = F.interpolate(features_bottleneck, size=features_skip_4x.shape[2:], mode='bilinear', align_corners=False)
        # DCNN 1x1 convolution (only necessary if the no. of channels should be increased from 64 to 256). 
        # features_dcnn = self.dcnn_conv(features_skip_4x)
        # Tensor concatination. 
        concat_tensor = torch.cat((features_4x, features_skip_4x), 1)
        # 3x3 Convolution
        predictions_4x = self.conv_3x3(concat_tensor)
        
        return predictions_4x, concat_tensor

""" Task 3. Second decoder of the SA module, simple 3x3 conv. """

class DecoderDeeplabSA(torch.nn.Module):
    def __init__(self, num_in_ch, num_out_ch):
        super(DecoderDeeplabSA, self).__init__()
        # 3x3 on concatenated feature maps. 
        self.conv_3x3 = torch.nn.Conv2d(num_in_ch, num_out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, features_4x_decoder, features_4x_sa):
        """
        DeepLabV3+ style decoder
        :param features_4x_decoder: features of encoder of scale == 4
        :param features_4x_sa: features of the sa module, same resolution as features_4x_decoder
        :return: final tensor of predictions
        """
        features_4x_decoder_sa = features_4x_decoder + features_4x_sa
        return self.conv_3x3(features_4x_decoder_sa)

""" Task 3. Second decoder of the SA module, deeper version with upscaling """

class DecoderConvUpscaleLayer(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

class DecoderDeeplabSADeeper(torch.nn.Module):
    def __init__(self, num_in_ch, num_out_ch):
        super(DecoderDeeplabSADeeper, self).__init__()
        # 3x3 conv. with upscaling and channel reduction.
        self.conv_upscale1 = DecoderConvUpscaleLayer(num_in_ch, int(num_in_ch/2), kernel_size=3, stride=1, padding=1)
        self.conv_upscale2 = DecoderConvUpscaleLayer(int(num_in_ch/2), int(num_in_ch/4), kernel_size=3, stride=1, padding=1)
        # 3x3 final conv. 
        self.conv_3x3 = torch.nn.Conv2d(int(num_in_ch/4), num_out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, features_4x_decoder, features_4x_sa, output_resolution):
        """
        DeepLabV3+ style decoder modified according the pad-net paper. 
        The input layer is run through a 3x3 conv. and then upscaled by a factor of 2 while 
        reducing the channel size by a factor of 2. This is done twice. Therefore, the output
        resolution is reached. 
        Finally, a 3x3 conv. is run decreasing the channel output size to the target output size. 
        :param features_4x_decoder: features of encoder of scale == 4
        :param features_4x_sa: features of the sa module, same resolution as features_4x_decoder
        :output_resolution: 2 tuple of ints (height, width)
        :return: final tensor of predictions
        """
        # Add the inputs from the first decoder and self attention module element-wise. 
        features_4x_decoder_sa = features_4x_decoder + features_4x_sa
        # Resolution for the first 2x upscaling. 
        intermediate_resolution = (int(output_resolution[0]/2), int(output_resolution[1]/2))
        # 3x3 conv and 2x upscaling. 
        out = self.conv_upscale1(features_4x_decoder_sa)
        out = F.interpolate(out, size=intermediate_resolution, mode='bilinear', align_corners=False)
        out = self.conv_upscale2(out)
        out = F.interpolate(out, size=output_resolution, mode='bilinear', align_corners=False)
        # Final 3x3 conv. layer (note: no ReLu and Batchnorm anymore -> final predictions).
        out = self.conv_3x3(out)
        return out

""" Self attention module. """

class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.attention = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # Zero weights of the attention layer. 
        with torch.no_grad():
            self.attention.weight.copy_(torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask

class SqueezeAndExcitation(torch.nn.Module):
    """
    Squeeze and excitation module as explained in https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channels, r=16):
        super(SqueezeAndExcitation, self).__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // r),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // r, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        N, C, H, W = x.shape
        squeezed = torch.mean(x, dim=(2, 3)).reshape(N, C)
        squeezed = self.transform(squeezed).reshape(N, C, 1, 1)
        return x * squeezed
