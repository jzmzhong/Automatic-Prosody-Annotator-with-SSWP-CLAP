# Multi-modal Automatic Prosody Annotation with Contrastive Pretraining of SSWP

<!-- ### [Arxiv](https://arxiv.org/abs/2309.05423) &nbsp;&nbsp;&nbsp;&nbsp; [Demo](https://jzmzhong.github.io/Automatic-Prosody-Annotator-With-SSWP-CLAP/) -->
<table style="border:1px solidb ack;margin-left:auto;margin-right:auto;border-spacing:20px;">
    <tr>
    <td><h3><a href="https://arxiv.org/abs/2309.05423">Arxiv</a></h3></td>
    <td><h3><a href="https://jzmzhong.github.io/Automatic-Prosody-Annotator-With-SSWP-CLAP">Demo</a></h3></td>
    <td><h3><a href="./">Code (Here)</a></h3></td>
    </tr>
</table>

# Model Architecture

<!-- Method -->
<h3>The Architecture of Our Proposed Work</h3>
<figure>
<p style="text-align:center"><img src="./figs/model_v2.3_trimmed.png" alt="AVSE" height=500px/></p>
</figure>

# Results & Demos

<!-- Ojective Evaluation -->
<h3>Objective Evaluation</h3>
The results of our proposed work, compared with previous benchmarks, are shown below.
<figure>
<p style="text-align:center"><img src="./figs/objective.png" alt="AVSE" height=140px/></p>
</figure>


### Audio Samples Demo are avaialble at: [Demo](https://jzmzhong.github.io/Automatic-Prosody-Annotator-With-SSWP-CLAP/)


# Quickstart

## Environment Installation

```bash
conda create -n clap python=3.10
conda activate clap
# you can also install pytorch by following the official instruction (https://pytorch.org/get-started/locally/)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Multi-modal Prosody Annotation

### Data Process

The model requires the input data to be aligned using Kaldi. After that, it needs to be converted to the format of the sample data.

Text Features are stroed in *.json; Audio features are stored in *.mel.npy.

```shell
./sample_data/prosody_annotation
├── wordboundarylevel_ling
│   └── internal-spk1-test
└── wordboundarylevel_mel
    └── internal-spk1-test
```

To convert audio to mel, you can refer to the following command:

```base
cd data_process
python 01_wav2mel.py
```

### Sample Inference Script

```base
bash ./example/01_inference_prosody_annotation.sh
```
### Released Multi-modal Prosody Annotator 

```
./released_model/pretrained_SSWP_CLAP.pt
```


# References

- [LAION-AI's CLAP](https://github.com/LAION-AI/CLAP)
- [CLAPSpeech](https://clapspeech.github.io/)
