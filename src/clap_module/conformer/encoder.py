# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import logging

import torch
import math
from .convolution import ConvolutionModule
from .encoder_layer import EncoderLayer
from .modules import get_activation
from .modules import VGG2L
from .modules import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from .embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from .modules import LayerNorm
from .multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from .modules import (
    PositionwiseFeedForward,
)
from .modules import repeat
from .sub_sampling import Conv2dSubsampling

from ..feature_fusion import AttentionPool1d, DAF, AFF, iAFF


class Encoder(torch.nn.Module):
    """Conformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Encoder positional encoding layer type.
        selfattention_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        ffn_layer_type="linear",
        ffn_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        activation_type="relu",
        use_cnn_module=True,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
        ctc_softmax=None,
        conditioning_layer_dim=None,
        max_seq_len=100,
        enable_fusion=False,
        fusion_type="",
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        self.max_seq_len = max_seq_len
        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            logging.info("encoder self-attention layer type = relative self-attention")
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        # feed-forward module definition
        if ffn_layer_type == "linear":
            ffn_layer = PositionwiseFeedForward
            ffn_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif ffn_layer_type == "conv1d":
            ffn_layer = MultiLayeredConv1d
            ffn_layer_args = (
                attention_dim,
                linear_units,
                ffn_conv_kernel_size,
                dropout_rate,
            )
        elif ffn_layer_type == "conv1d-linear":
            ffn_layer = Conv1dLinear
            ffn_layer_args = (
                attention_dim,
                linear_units,
                ffn_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                ffn_layer(*ffn_layer_args),
                ffn_layer(*ffn_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers
        self.use_conditioning = True if ctc_softmax is not None else False
        if self.use_conditioning:
            self.ctc_softmax = ctc_softmax
            self.conditioning_layer = torch.nn.Linear(
                conditioning_layer_dim, attention_dim
            )
        
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        if (self.enable_fusion) and (self.fusion_type in ['daf_1d','aff_1d','iaff_1d']):
            raise NotImplementedError
            if self.fusion_type == 'daf_1d':
                self.fusion_model = DAF()
            elif self.fusion_type == 'aff_1d':
                self.fusion_model = AFF(channels=attention_dim, type='1D')
            elif self.fusion_type == 'iaff_1d':
                self.fusion_model = iAFF(channels=attention_dim, type='1D')
        elif (self.enable_fusion) and (self.fusion_type in ['attnpool_1d']):
            self.attnpool = AttentionPool1d(max_seq_len, attention_dim, attention_heads)

    def forward(self, audio_features, mixup_lambda=None, device="cuda"):
        """Encode input sequence.

        Args:
            audio_features: Input dict {"mel": xs, "mel_mask": masks}
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, 1, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, 1, time).

        """

        xs, masks = audio_features["mel"], audio_features["mel_mask"]
        xs, masks = xs.to(device=device, non_blocking=True), masks.to(device=device, non_blocking=True)
        input_len = xs.shape[1]
        max_len = min(input_len, self.max_seq_len)
        xs=xs[:, :max_len, :]
        masks = masks[:, :, :max_len]

        if isinstance(self.embed, (Conv2dSubsampling, VGG2L)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks) 
            intermediate_outputs = None
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    # intermediate branches also require normalization.
                    encoder_output = xs
                    if isinstance(encoder_output, tuple):
                        encoder_output = encoder_output[0]

                    if self.normalize_before:
                        encoder_output = self.after_norm(encoder_output)

                    intermediate_outputs.append(encoder_output)

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)

                        if isinstance(xs, tuple):
                            x, pos_emb = xs[0], xs[1]
                            x = x + self.conditioning_layer(intermediate_result)
                            xs = (x, pos_emb)
                        else:
                            xs = xs + self.conditioning_layer(intermediate_result)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)
        
        if (self.enable_fusion) and (self.fusion_type in ['daf_1d','aff_1d','iaff_1d']):
            raise NotImplementedError
        elif (self.enable_fusion) and (self.fusion_type in ['attnpool_1d']):
            # import pdb; pdb.set_trace()
            xs = self.attnpool(xs)
            
        return {
            'embedding': xs,
            'masks': masks,
            'intermediate_outputs': intermediate_outputs
        }
