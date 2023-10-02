"""
Task 2
Branched architecture. 
"""
import torch
import torch.nn.functional as F
from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p

class ModelDeepLabV3PlusBranched(torch.nn.Module):
    
    def __init__(self, cfg, outputs_desc):
        super().__init__()

        self.outputs_desc = outputs_desc
        ch_out_segm = outputs_desc['semseg']
        ch_out_depth = outputs_desc['depth']

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
        
        # Semantic segmentation head. 
        self.aspp_segm = ASPP(ch_out_encoder_bottleneck, 256)
        self.decoder_segm = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_segm)

        # Depth estimation head. 
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)
        self.decoder_depth = DecoderDeeplabV3p(256, ch_out_encoder_4x, ch_out_depth)
    
    def forward(self, x):
        input_resolution = (x.shape[2], x.shape[3]) # Height, width
        
        # Encoder. 
        features = self.encoder(x)

        # Uncomment to see the scales of feature pyramid with their respective number of channels.
        # print(", ".join([f"{k}:{v.shape[1]}" for k, v in features.items()]))

        # Get features from encoder module. 
        lowest_scale = max(features.keys())
        features_lowest = features[lowest_scale]

        # Branched network decoders. 
        
        # Segmentation head. 
        features_segm = self.aspp_segm(features_lowest)
        predictions_4x_segm, _ = self.decoder_segm(features_segm, features[4])
        predictions_1x_segm = F.interpolate(predictions_4x_segm, size=input_resolution, mode='bilinear', align_corners=False)

        # Depth estimation head. 
        features_depth = self.aspp_depth(features_lowest)
        predictions_4x_depth, _ = self.decoder_depth(features_depth, features[4])
        predictions_1x_depth = F.interpolate(predictions_4x_depth, size=input_resolution, mode='bilinear', align_corners=False)

        # Return predictions. 
        out = {
            'semseg': predictions_1x_segm,
            'depth': predictions_1x_depth,
        }
        return out
