""" CLAP Model

Adapted from CLIP: https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
Adapted to the Audio Task.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import logging

from .pann_model import create_pann_model
from .htsat import create_htsat_model
from .conformer.conformer_model import create_conformer_model

from transformers import BertModel, RobertaModel, BartModel, AutoConfig, HubertModel

from .feature_fusion import AttentionPool1d


class MLPLayers(nn.Module):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16.
    """

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
            self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


# Audio Config Class
@dataclass
class CLAPAudioCfp:
    model_type: str = "HTSAT"
    model_name: str = "base"
    pretrained_audio: str = ""
    # Param
    audio_length: int = 1024
    clip_samples: int = 480000
    mel_bins: int = 64
    sample_rate: int = 48000
    window_size: int = 1024
    hop_size: int = 1024
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527


@dataclass
class CLAPConformerAudioCfp:
    model_type: str = "Conformer"
    model_name: str = "base"
    pretrained_audio: str = ""
    # Param
    mel_bins: int = 80
    max_time_bins: int = 100
    attn_dim: int = 256
    layers: int = 4
    units: int = 1024
    heads: int = 4
    pos_enc_layer_type: str = "rel_pos"
    ffn_layer_type: str = "linear"
    ffn_conv_kernel_size: int = 0
    use_macaron_style_in_conformer: bool = True
    self_attn_layer_type: str = "rel_selfattn"
    activation_type: str = "swish"
    dropout_rate: float = 0.2
    pos_dropout_rate: float = 0.2
    attn_dropout_rate: float = 0.2


@dataclass
class CLAPHuBERTAudioCfp:
    model_type: str = "hubert"
    model_name: str = "large-ls960-ft"
    pretrained_audio: str = ""
    frame_rate: float = 0.02
    sampling_rate: int = 16000
    hidden_size: int = 1024
    context_time_bins: int = 0
    max_time_bins: int = 50
    heads: int = 16
    load_pretrained_weights: bool = True


@dataclass
class CLAPTextCfg:
    model_type: str = "bert"
    model_name: str = "base-uncased"
    pretrained_text: str = ""
    context_length: int = 77
    max_num_subword: int = 5
    vocab_size: int = 49408
    width: int = 768
    heads: int = 8
    layers: int = 12
    load_pretrained_weights: bool = True


class CLAP(nn.Module):
    def __init__(
            self,
            args,
            joint_embed_shape: int,
            audio_cfg: CLAPAudioCfp,
            text_cfg: CLAPTextCfg,
            enable_fusion: bool = False,
            fusion_type: str = 'None',
            mlp_act: str = 'relu',
            quick_gelu: bool = False,
    ):

        super().__init__()
        if isinstance(audio_cfg, dict):
            if audio_cfg["model_type"] in ("Conformer"):
                audio_cfg = CLAPConformerAudioCfp(**audio_cfg)
            elif audio_cfg["model_type"] in ("hubert"):
                audio_cfg = CLAPHuBERTAudioCfp(**audio_cfg)
            else:
                audio_cfg = CLAPAudioCfp(**audio_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLAPTextCfg(**text_cfg)

        self.data_filling = args.data_filling
        self.data_truncating = args.data_truncating

        self.joint_embed_shape = joint_embed_shape
        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.mlp_act = mlp_act
        self.context_length = text_cfg.context_length

        # set activation of clip text encoder
        """OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        memory efficient in recent PyTorch releases (>= 1.10).
        NOTE: timm models always use native GELU regardless of quick_gelu flag."""
        act_layer = QuickGELU if quick_gelu else nn.GELU

        # set activation of MLP
        if mlp_act == 'relu':
            mlp_act_layer = nn.ReLU()
        elif mlp_act == 'gelu':
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError

        # audio encoder
        if audio_cfg.model_type == "PANN":
            self.audio_branch = create_pann_model(audio_cfg, enable_fusion, fusion_type)
        elif audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(audio_cfg, enable_fusion, fusion_type)
        elif audio_cfg.model_type == "Conformer":
            self.audio_branch = create_conformer_model(audio_cfg, enable_fusion, fusion_type)
        elif audio_cfg.model_type == "hubert":
            # hubert model
            if audio_cfg.load_pretrained_weights:
                config = AutoConfig.from_pretrained("facebook/hubert-{}".format(audio_cfg.model_name))
                config.mask_time_prob = 0.
                # config.mask_time_length = 1
                config.mask_feature_prob = 0.
                # config.mask_feature_length = 1
                self.audio_branch = HubertModel.from_pretrained("facebook/hubert-{}".format(audio_cfg.model_name),
                                                                config=config)
            else:
                config = AutoConfig.from_pretrained("facebook/hubert-{}".format(audio_cfg.model_name))
                self.audio_branch = HubertModel(config)
            # attentive pooling
            if self.enable_fusion and (self.fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']):
                raise NotImplementedError
            elif self.enable_fusion and (self.fusion_type in ['attnpool_1d']):
                self.frames2frame = AttentionPool1d(audio_cfg.max_time_bins, audio_cfg.hidden_size, audio_cfg.heads)
            else:
                self.frames2frame = None
        else:
            logging.error(f"Model config for {audio_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {audio_cfg.model_type} not found.")
        try:
            self.audio_projection = nn.Sequential(
                nn.Linear(audio_cfg.attn_dim, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
            )
        except:
            self.audio_projection = nn.Sequential(
                nn.Linear(audio_cfg.hidden_size, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
            )
        self.audio_transform = MLPLayers(units=[self.joint_embed_shape,
                                                self.joint_embed_shape,
                                                self.joint_embed_shape], dropout=0.1)
        self.audio_branch_type = audio_cfg.model_type

        # text encoder
        if text_cfg.model_type == "transformer":
            self.text_branch = Transformer(
                width=text_cfg.width,
                layers=text_cfg.layers,
                heads=text_cfg.heads,
                act_layer=act_layer,
            )
            self.vocab_size = text_cfg.vocab_size
            self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
            self.positional_embedding = nn.Parameter(
                torch.empty(self.context_length, text_cfg.width)
            )
            self.ln_final = LayerNorm(text_cfg.width)
        elif text_cfg.model_type in ("bert", "roberta", "bart"):
            if text_cfg.model_type == "bert":
                if text_cfg.load_pretrained_weights:
                    try:
                        self.text_branch = BertModel.from_pretrained("bert-{}".format(text_cfg.model_name))
                    except:
                        self.text_branch = BertModel.from_pretrained("prajjwal1/bert-{}".format(text_cfg.model_name))
                else:
                    try:
                        config = AutoConfig.from_pretrained("bert-{}".format(text_cfg.model_name))
                    except:
                        config = AutoConfig.from_pretrained("prajjwal1/bert-{}".format(text_cfg.model_name))
                    self.text_branch = BertModel(config)
            elif text_cfg.model_type == "roberta":
                self.text_branch = RobertaModel.from_pretrained('roberta-base')
            elif text_cfg.model_type == "bart":
                self.text_branch = BartModel.from_pretrained('facebook/bart-base')
            else:
                raise NotImplementedError
        else:
            logging.error(f"Model config for {text_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {text_cfg.model_type} not found.")
        if self.enable_fusion and (self.fusion_type in ['daf_1d', 'aff_1d', 'iaff_1d']):
            raise NotImplementedError

        elif self.enable_fusion and (self.fusion_type in ['attnpool_1d']):
            self.subword2word = AttentionPool1d(text_cfg.max_num_subword, text_cfg.width, text_cfg.heads)
        else:
            self.subword2word = None
        self.text_projection = nn.Sequential(
            nn.Linear(text_cfg.width, self.joint_embed_shape),
            mlp_act_layer,
            nn.Linear(self.joint_embed_shape, self.joint_embed_shape)
        )
        self.text_transform = MLPLayers(units=[self.joint_embed_shape,
                                               self.joint_embed_shape,
                                               self.joint_embed_shape], dropout=0.1)
        self.text_branch_type = text_cfg.model_type

        self.logit_scale_a = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_t = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer("attn_mask", self.build_attention_mask(), persistent=False)

        self.init_text_branch_parameters()

    def init_text_branch_parameters(self):
        if self.text_branch_type == "transformer":
            nn.init.normal_(self.token_embedding.weight, std=0.02)
            nn.init.normal_(self.positional_embedding, std=0.01)
            proj_std = (self.text_branch.width ** -0.5) * (
                    (2 * self.text_branch.layers) ** -0.5
            )
            attn_std = self.text_branch.width ** -0.5
            fc_std = (2 * self.text_branch.width) ** -0.5
            for block in self.text_branch.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_branch_type in ("bert", "roberta"):
            width = self.text_branch.embeddings.word_embeddings.weight.shape[-1]
        elif self.text_branch_type == "bart":
            width = self.text_branch.shared.weight.shape[-1]
        else:
            width = self.text_branch.width
        nn.init.constant_(self.logit_scale_a, np.log(1 / 0.07))
        nn.init.constant_(self.logit_scale_t, np.log(1 / 0.07))

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_audio(self, audio, device):
        if self.audio_branch_type == "hubert":
            if "sent_wavs" in audio:
                xs = pad_sequence([torch.from_numpy(x) for x in audio["sent_wavs"]], batch_first=True,
                                  padding_value=0.).to(device=device)
                xs = self.audio_branch(input_values=xs, output_hidden_states=False,
                                       return_dict=True)  # mix lambda needs to add
                xs = xs["last_hidden_state"]
                # select the word-aligned audio latents from the sequence, pad to certain length and truncate if necessary
                new_xs = []
                for x, start_end in zip(xs, audio["token_indices"]):
                    start, end = int(start_end[0] / self.audio_cfg.frame_rate), int(
                        start_end[1] / self.audio_cfg.frame_rate)
                    assert start < end, (start, end, x)
                    assert end <= len(x), (start, end, x)
                    x = x[start:end, :]
                    if x.shape[0] > self.audio_cfg.max_time_bins:
                        if self.data_truncating == "front_trunc":
                            x = x[:self.audio_cfg.max_time_bins, :]
                        elif self.data_truncating == "back_trunc":
                            x = x[-self.audio_cfg.max_time_bins:, :]
                        elif self.data_truncating == "cent_trunc":
                            x = x[int(0.5 * (x.shape[0] - self.audio_cfg.max_time_bins)): int(
                                0.5 * (x.shape[0] + self.audio_cfg.max_time_bins)), :]
                        else:
                            raise NotImplementedError
                    new_xs.append(x)
                if self.data_filling == "pad":
                    new_xs.append(torch.ones((self.audio_cfg.max_time_bins, new_xs[-1].shape[1]), dtype=float))
                    new_xs = pad_sequence(new_xs, batch_first=True, padding_value=0.)[:-1, :, :]
                else:
                    raise NotImplementedError
            else:
                xs = pad_sequence([torch.from_numpy(x) for x in audio["token_wavs"]], batch_first=True,
                                  padding_value=0.).to(device=device)
                xs = self.audio_branch(input_values=xs, output_hidden_states=False,
                                       return_dict=True)  # mix lambda needs to add
                xs = xs["last_hidden_state"]
                # pad to certain length and truncate if necessary
                new_xs = []
                for x in xs:
                    if x.shape[0] > self.audio_cfg.max_time_bins:
                        if self.data_truncating == "front_trunc":
                            x = x[:self.audio_cfg.max_time_bins, :]
                        elif self.data_truncating == "back_trunc":
                            x = x[-self.audio_cfg.max_time_bins:, :]
                        elif self.data_truncating == "cent_trunc":
                            x = x[int(0.5 * (x.shape[0] - self.audio_cfg.max_time_bins)): int(
                                0.5 * (x.shape[0] + self.audio_cfg.max_time_bins)), :]
                        else:
                            raise NotImplementedError
                    new_xs.append(x)
                if self.data_filling == "pad":
                    new_xs.append(torch.ones((self.audio_cfg.max_time_bins, new_xs[-1].shape[1]), dtype=float))
                    new_xs = pad_sequence(new_xs, batch_first=True, padding_value=0.)[:-1, :, :]
                else:
                    raise NotImplementedError
            if self.frames2frame:
                x = self.frames2frame(new_xs)
                x = self.audio_projection(x)
        else:
            x = self.audio_branch(audio, mixup_lambda=None, device=device)  # mix lambda needs to add
            x = self.audio_projection(x["embedding"])
        return x

    def encode_text(self, text, device):
        if self.text_branch_type == "transformer":
            text = text.to(device=device, non_blocking=True)
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.text_branch(x, attn_mask=self.attn_mask)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = self.text_projection(x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        elif self.text_branch_type == "bert":
            if self.subword2word:
                xs = self.text_branch(
                    input_ids=text["input_ids"].to(
                        device=device, non_blocking=True
                    ),
                    attention_mask=text["attention_mask"].to(
                        device=device, non_blocking=True
                    ),
                    token_type_ids=text["token_type_ids"].to(
                        device=device, non_blocking=True
                    ),
                )["last_hidden_state"]
                # import pdb; pdb.set_trace()
                # select the subwords from the sequence, pad to certain length and truncate if necessary
                new_xs = []
                for x, start_end in zip(xs, text["token_indices"]):
                    x = x[int(start_end[0]):int(start_end[1]), :]
                    if x.shape[0] > self.text_cfg.max_num_subword:
                        if self.data_truncating == "front_trunc":
                            x = x[:self.text_cfg.max_num_subword, :]
                        elif self.data_truncating == "back_trunc":
                            x = x[-self.text_cfg.max_num_subword:, :]
                        elif self.data_truncating == "cent_trunc":
                            x = x[int(0.5 * (x.shape[0] - self.text_cfg.max_num_subword)): int(
                                0.5 * (x.shape[0] + self.text_cfg.max_num_subword)), :]
                        else:
                            raise NotImplementedError
                    new_xs.append(x)
                if self.data_filling == "pad":
                    new_xs.append(torch.ones((self.text_cfg.max_num_subword, new_xs[-1].shape[1]), dtype=float))
                    new_xs = pad_sequence(new_xs, batch_first=True, padding_value=0.)[:-1, :, :]
                else:
                    raise NotImplementedError
                x = self.subword2word(new_xs)
                x = self.text_projection(x)
            else:
                x = self.text_branch(
                    input_ids=text["input_ids"].to(
                        device=device, non_blocking=True
                    ),
                    attention_mask=text["attention_mask"].to(
                        device=device, non_blocking=True
                    ),
                    token_type_ids=text["token_type_ids"].to(
                        device=device, non_blocking=True
                    ),
                )["pooler_output"]
                x = self.text_projection(x)
        elif self.text_branch_type == "roberta":
            x = self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["pooler_output"]
            x = self.text_projection(x)
        elif self.text_branch_type == "bart":
            x = torch.mean(self.text_branch(
                input_ids=text["input_ids"].to(device=device, non_blocking=True),
                attention_mask=text["attention_mask"].to(
                    device=device, non_blocking=True
                ),
            )["encoder_last_hidden_state"], axis=1)
            x = self.text_projection(x)
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")
        return x

    def encode_text_sent(self, texts, device):
        if self.text_branch_type == "bert":
            if self.subword2word:
                xs = self.text_branch(
                    input_ids=text["input_ids"].to(
                        device=device, non_blocking=True
                    ),
                    attention_mask=text["attention_mask"].to(
                        device=device, non_blocking=True
                    ),
                    token_type_ids=text["token_type_ids"].to(
                        device=device, non_blocking=True
                    ),
                )["last_hidden_state"]
                # import pdb; pdb.set_trace()
                # select the subwords from the sequence, pad to certain length and truncate if necessary
                new_xs = []
                for x, start_end in zip(xs, text["token_indices"]):
                    x = x[int(start_end[0]):int(start_end[1]), :]
                    if x.shape[0] > self.text_cfg.max_num_subword:
                        if self.data_truncating == "front_trunc":
                            x = x[:self.text_cfg.max_num_subword, :]
                        elif self.data_truncating == "back_trunc":
                            x = x[-self.text_cfg.max_num_subword:, :]
                        elif self.data_truncating == "cent_trunc":
                            x = x[int(0.5 * (x.shape[0] - self.text_cfg.max_num_subword)): int(
                                0.5 * (x.shape[0] + self.text_cfg.max_num_subword)), :]
                        else:
                            raise NotImplementedError
                    new_xs.append(x)
                if self.data_filling == "pad":
                    new_xs.append(torch.ones((self.text_cfg.max_num_subword, new_xs[-1].shape[1]), dtype=float))
                    new_xs = pad_sequence(new_xs, batch_first=True, padding_value=0.)[:-1, :, :]
                else:
                    raise NotImplementedError
                x = self.subword2word(new_xs)
                x = self.text_projection(x)
            else:
                raise NotImplementedError
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")
        return x

    def forward(self, audio, text, device=None):
        """Forward audio and text into the CLAP

        Args:
            audio (torch.Tensor): (batch_size, audio_length) the time-domain audio input / the batch of 
                                   mel_spec and longer list.
            text (torch.Tensor): the text token input
            device (str, optional): device. Defaults to None.

        Returns:
            audio_features (torch.Tensor): (batch_size, audio_feature_dim)
            text_features (torch.Tensor): (batch_size, text_feature_dim)
            audio_features_mlp (torch.Tensor): (batch_size, audio_feature_dim)
            text_features_mlp (torch.Tensor): (batch_size, text_feature_dim)
        """
        if device is None:
            if audio is not None:
                device = audio.device
            elif text is not None:
                device = text.device
        if audio is None and text is None:
            # a hack to get the logit scale
            return self.logit_scale_a.exp(), self.logit_scale_t.exp()
        elif audio is None:
            return self.encode_text(text, device=device)
        elif text is None:
            return self.encode_audio(audio, device=device)

        audio_features = self.encode_audio(audio, device=device)
        audio_features = F.normalize(audio_features, dim=-1)
        text_features = self.encode_text(text, device=device)
        text_features = F.normalize(text_features, dim=-1)

        audio_features_mlp = self.audio_transform(audio_features)
        text_features_mlp = self.text_transform(text_features)
        # Four outputs: audio features (basic & MLP), text features (basic & MLP)
        return (
            audio_features,
            text_features,
            audio_features_mlp,
            text_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )

    def forward_sent(self, audios, texts, device=None):
        """Forward audio and text into the CLAP

        Args:
            audios (torch.Tensor): (batch_size, audio_length) the time-domain audio input / the batch of mel_spec and longer list.
            texts (torch.Tensor): the text token input
            device (str, optional): device. Defaults to None.
        """
        if device is None:
            if audios is not None:
                device = audios.device
            elif texts is not None:
                device = texts.device
        if audios is None and texts is None:
            # a hack to get the logit scale
            return self.logit_scale_a.exp(), self.logit_scale_t.exp()
        elif audios is None:
            return self.encode_text(texts, device=device)
        elif texts is None:
            return self.encode_audio(audios, device=device)

        audio_features = self.encode_audio(audios, device=device)
        audio_features = F.normalize(audio_features, dim=-1)
        text_features = self.encode_text(texts, device=device)
        text_features = F.normalize(text_features, dim=-1)

        audio_features_mlp = self.audio_transform(audio_features)
        text_features_mlp = self.text_transform(text_features)
        # Four outputs: audio features (basic & MLP), text features (basic & MLP)
        return (
            audio_features,
            text_features,
            audio_features_mlp,
            text_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )

    def get_logit_scale(self):
        return self.logit_scale_a.exp(), self.logit_scale_t.exp()

    def get_text_embedding(self, data):
        """Get the text embedding from the model

        Args:
            data (torch.Tensor): a tensor of text embedding

        Returns:
            text_embed (torch.Tensor): a tensor of text_embeds (N, D)
        """
        device = next(self.parameters()).device
        for k in data:
            data[k] = data[k].to(device)
        text_embeds = self.encode_text(data, device=device)
        text_embeds = F.normalize(text_embeds, dim=-1)

        return text_embeds

    def get_audio_embedding(self, data):
        """Get the audio embedding from the model

        Args:
            data (list): a list of dict the audio input dict list from 'get_audio_feature' method

        Returns:
            audio_embed (torch.Tensor): a tensor of audio_embeds (N, D)
        """
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        audio_embeds = self.encode_audio(input_dict, device=device)["embedding"]
        audio_embeds = self.audio_projection(audio_embeds)
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds

    def audio_infer(self, audio, hopsize=None, device=None):
        """Forward one audio and produce the audio embedding

        Args:
            audio (audio_length): the time-domain audio input, notice that it must be only one input
            hopsize (int, optional):  the overlap hopsize as the sliding window. Defaults to None.
            device (str, optional): device. Defaults to None.

        Returns:
            output_dict ({
            key: [n, (embedding_shape)] if "HTS-AT"
            or
            key: [(embedding_shape)] if "PANN"
        }): the list of key values of the audio branch
        """

        assert not self.training, "the inference mode must be run at eval stage"
        output_dict = {}
        # PANN
        if self.audio_cfg.model_type == "PANN":
            audio_input = audio.unsqueeze(dim=0)
            output_dict[key] = self.encode_audio(audio_input, device=device)[key].squeeze(dim=0)
        elif self.audio_cfg.model_type == "HTSAT":
            # repeat
            audio_len = len(audio)
            k = self.audio_cfg.clip_samples // audio_len
            if k > 1:
                audio = audio.repeat(k)
                audio_len = len(audio)

            if hopsize is None:
                hopsize = min(hopsize, audio_len)

            if audio_len > self.audio_cfg.clip_samples:
                audio_input = [
                    audio[pos: pos + self.audio_cfg.clip_samples].clone()
                    for pos in range(
                        0, audio_len - self.audio_cfg.clip_samples, hopsize
                    )
                ]
                audio_input.append(audio[-self.audio_cfg.clip_samples:].clone())
                audio_input = torch.stack(audio_input)
                output_dict[key] = self.encode_audio(audio_input, device=device)[key]
            else:
                audio_input = audio.unsqueeze(dim=0)
                output_dict[key] = self.encode_audio(audio_input, device=device)[key].squeeze(dim=0)

        return output_dict


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16
    """

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
