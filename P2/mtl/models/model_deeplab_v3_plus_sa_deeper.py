"""
Task 3
Deeper second decoder. Two 3x3 convs with 2x upscaling and 2x channel size reduction.
Plus one final 3x3 convolution. 
"""
import torch
import torch.nn.functional as F
from mtl.models.model_parts import Encoder, get_encoder_channel_counts, ASPP, DecoderDeeplabV3p
from mtl.models.model_parts import DecoderDeeplabV3pSA, SelfAttention, DecoderDeeplabSA, DecoderDeeplabSADeeper

class ModelDeepLabV3PlusBranchedSADeeper(torch.nn.Module):
    
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
        # Decoder of the first network part, combine ASPP output with feature map from shared encoder. 
        self.decoder_1_segm = DecoderDeeplabV3pSA(256, ch_out_encoder_4x, ch_out_segm)
        # Self-attention (SA) module. 
        self.sa_segm = SelfAttention(256+ch_out_encoder_4x, 256+ch_out_encoder_4x)
        # Decoder of the second network part, combine SA depth output with decoder_skip_segm output.
        self.decoder_2_segm = DecoderDeeplabSADeeper(256+ch_out_encoder_4x, ch_out_segm)

        # Depth estimation head. 
        self.aspp_depth = ASPP(ch_out_encoder_bottleneck, 256)
        # Decoder of the first network part, combine ASPP output with feature map from shared encoder.
        self.decoder_1_depth = DecoderDeeplabV3pSA(256, ch_out_encoder_4x, ch_out_depth)
        # Self-attention (SA) module. 
        self.sa_depth = SelfAttention(256+ch_out_encoder_4x, 256+ch_out_encoder_4x)
        # Decoder of the second network part, combine SA segm output with decoder_skip_depth output. 
        self.decoder_2_depth = DecoderDeeplabSADeeper(256+ch_out_encoder_4x, ch_out_depth)
    
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

        # First network part: shared encoder, and split ASPP, decoder, and SA modules.         
        # Segmentation head. 
        features_segm = self.aspp_segm(features_lowest)
        predictions_4x_segm_1, features_4x_segm = self.decoder_1_segm(features_segm, features[4])
        predictions_1x_segm_1 = F.interpolate(predictions_4x_segm_1, size=input_resolution, mode='bilinear', align_corners=False)
        sa_features_segm = self.sa_segm(features_4x_segm)
        # Depth estimation head. 
        features_depth = self.aspp_depth(features_lowest)
        predictions_4x_depth_1, features_4x_depth = self.decoder_1_depth(features_depth, features[4])
        predictions_1x_depth_1 = F.interpolate(predictions_4x_depth_1, size=input_resolution, mode='bilinear', align_corners=False)
        sa_features_depth = self.sa_depth(features_4x_depth)

        # Second network part: Decoder (combines SA features + first network part features) and final prediction. 
        # Segmentation head.
        predictions_1x_segm_2 = self.decoder_2_segm(features_4x_segm, sa_features_depth, input_resolution)
        # Depth estimation head.
        predictions_1x_depth_2 = self.decoder_2_depth(features_4x_depth, sa_features_segm, input_resolution)

        # Return predictions.
        # Values are lists of both network part predictions. The loss is computed for both predictions individually.
        # The losses are then summed and averaged for a final loss value. 
        # See file experiment_semseg_with_depth.py, function training_step(self, batch, batch_nb). 
        out = {
            'semseg': [predictions_1x_segm_1, predictions_1x_segm_2],
            'depth': [predictions_1x_depth_1, predictions_1x_depth_2],
        }
        return out
