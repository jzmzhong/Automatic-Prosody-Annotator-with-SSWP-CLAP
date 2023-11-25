from .encoder import Encoder as ConformerEncoder


def create_conformer_model(audio_cfg, enable_fusion=False, fusion_type="None"):
    model = ConformerEncoder(
        idim=audio_cfg.mel_bins,
        attention_dim=audio_cfg.attn_dim,
        attention_heads=audio_cfg.heads,
        linear_units=audio_cfg.units,
        num_blocks=audio_cfg.layers,
        input_layer="linear",
        dropout_rate=audio_cfg.dropout_rate,
        positional_dropout_rate=audio_cfg.pos_dropout_rate,
        attention_dropout_rate=audio_cfg.attn_dropout_rate,
        normalize_before=True,
        concat_after=False,
        ffn_layer_type=audio_cfg.ffn_layer_type,
        ffn_conv_kernel_size=audio_cfg.ffn_conv_kernel_size,
        macaron_style=audio_cfg.use_macaron_style_in_conformer,
        pos_enc_layer_type=audio_cfg.pos_enc_layer_type,
        selfattention_layer_type=audio_cfg.self_attn_layer_type,
        activation_type=audio_cfg.activation_type,
        use_cnn_module=True,
        cnn_module_kernel=7,
        zero_triu=False,
        enable_fusion=enable_fusion,
        fusion_type=fusion_type,
        max_seq_len=audio_cfg.max_time_bins
        )

    return model
