import torch
import torch.nn.functional as F
from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p

class ModelDeepLabV3Plus(torch.nn.Module):
    def __init__(self, cfg, outputs_desc):
        super().__init__()

        self.outputs_desc = outputs_desc
        ch_out = sum(outputs_desc.values())

        self.encoder = Encoder(
            cfg.model_encoder_name,
            # Option to use pretrained ResNet labels. 
            pretrained=True,
            zero_init_residual=True,
            # Option to use dilated conv.
            # ResNet layers (2, 3, 4). 
            replace_stride_with_dilation=(False, False, True),  
        )

        ch_out_encoder_bottleneck, ch_out_encoder_4x = get_encoder_channel_counts(cfg.model_encoder_name)

        self.aspp = ASPP(ch_out_encoder_bottleneck, 256)

        self.decoder = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out)

    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3]) # Height, width
        
        # Encoder. 
        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        # ASPP module. 
        lowest_scale = max(features.keys())
        features_lowest = features[lowest_scale] # Endoder output features.
        features_tasks = self.aspp(features_lowest)

        # Decoder. 
        # Provide ASPP output features and encoder feature maps (after the 2nd conv. layer) as arguments. 
        # Return predictions and features (suppressed). 
        predictions_4x, _ = self.decoder(features_tasks, features[4])
        # Up-sample the feature map to the original image input resolution (224x224).
        # align_corners=False to interpolate according to the pixel values instead of using equidistant
        # steps between known pixel values as is the case for align_corners=True. 
        predictions_1x = F.interpolate(predictions_4x, size=input_resolution, mode='bilinear', align_corners=False)

        # Return predictions. 
        out = {}
        offset = 0
        for task, num_ch in self.outputs_desc.items():
            out[task] = predictions_1x[:, offset:offset+num_ch, :, :]
            offset += num_ch
            
        return out
