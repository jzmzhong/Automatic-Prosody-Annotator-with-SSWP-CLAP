import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class LinearProbe(nn.Module):
    def __init__(self, model, text_freeze, audio_freeze, in_ch, out_ch, audio_weight=1., classification="bilstm",
                 hier_labels=False, act="softmax"):
        """Linear probe module for CLAP model

        Args:
            model (nn.Module): the torch model
            text_freeze (bool): if True, then use the MLP layer as the linear probe module
            audio_freeze (bool): if Ture, then freeze all the CLAP model's audio encoder layers when training 
                                 the linear probe
            in_ch (int): the output channel from CLAP model
            out_ch (int): the output channel from linear probe (class_num)
            audio_weight (int, optional): the weight of audio latent features to text latent features. Defaults to 1..
            classification (str, optional): classification function. Defaults to "bilstm".
            hier_labels (bool, optional): if True, nn.Linear(in_ch, out_ch). Defaults to False.
            act (str, optional): the activation function before the loss function. Defaults to "softmax".

        """
        
        super().__init__()
        self.clap_model = model
        self.text_freeze = text_freeze
        self.audio_freeze = audio_freeze
        if classification == "bilstm":
            self.classification = nn.LSTM(in_ch,
                                          in_ch // 2,
                                          num_layers=1,
                                          batch_first=True,
                                          bidirectional=True)
        elif classification == "fc":
            self.classification = None
        else:
            raise NotImplementedError
        if not hier_labels:
            self.lp = nn.Linear(in_ch, out_ch)
        else:
            self.lp_l3 = nn.Linear(in_ch, 1)
            self.lp_l4 = nn.Linear(in_ch, 1)
            self.lp_l5 = nn.Linear(in_ch, 1)
        self.hier_labels = hier_labels
        self.audio_weight = audio_weight

        # import pdb; pdb.set_trace()
        if self.text_freeze:
            for param in self.clap_model.text_branch.parameters():
                param.requires_grad = False
            for param in self.clap_model.text_projection.parameters():
                param.requires_grad = False
            for param in self.clap_model.text_transform.parameters():
                param.requires_grad = False

        if self.audio_freeze:
            for param in self.clap_model.audio_branch.parameters():
                param.requires_grad = False
            for param in self.clap_model.audio_projection.parameters():
                param.requires_grad = False
            for param in self.clap_model.audio_transform.parameters():
                param.requires_grad = False

        if act == 'None':
            self.act = None
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'elu':
            self.act = nn.ELU()
        elif act == 'prelu':
            self.act = nn.PReLU(num_parameters=in_ch)
        elif act == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, texts, audios, mix_lambda=None, device=None):
        """
        Args:
            x: waveform, torch.tensor [batch, t_samples] / batch of mel_spec and longer list
            mix_lambda: torch.tensor [batch], the mixup lambda
        Returns:
            class_prob: torch.tensor [batch, class_num]

        """
        # batchnorm cancel grandient
        if self.text_freeze:
            self.clap_model.text_branch.eval()
            self.clap_model.text_projection.eval()
            self.clap_model.text_transform.eval()
        if self.audio_freeze:
            self.clap_model.audio_branch.eval()
            self.clap_model.audio_projection.eval()
            self.clap_model.audio_transform.eval()

        # extract sentences of word-level latents
        sentences_audio_featues = []
        sentences_text_featues = []
        for text, audio in zip(texts, audios):
            audio_features, text_features, _, _, _, _ = self.clap_model(audio, text, device=device)
            sentences_audio_featues.append(audio_features)
            sentences_text_featues.append(text_features)

        masks = pad_sequence(sentences_audio_featues, batch_first=True, padding_value=-999.)[:, :, :1] == -999.
        sentences_audio_featues = pad_sequence(sentences_audio_featues, batch_first=True, padding_value=0.)[:, :, :]
        sentences_text_featues = pad_sequence(sentences_text_featues, batch_first=True, padding_value=0.)[:, :, :]

        if self.classification:
            out = self.classification(
                (sentences_text_featues + self.audio_weight * sentences_audio_featues) / (1 + self.audio_weight))
        else:
            out = [(sentences_text_featues + self.audio_weight * sentences_audio_featues) / (1 + self.audio_weight)]
        if not self.hier_labels:
            out = self.lp(out[0])
            if self.act is not None:
                out = self.act(out)
            return out, masks
        else:
            out_l3, out_l4, out_l5 = self.lp_l3(out[0]), self.lp_l4(out[0]), self.lp_l5(out[0])
            return torch.cat([out_l3, out_l4, out_l5], dim=-1), masks


