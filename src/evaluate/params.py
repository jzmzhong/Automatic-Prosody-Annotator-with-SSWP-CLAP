import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


def parse_args():
    parser = argparse.ArgumentParser()

    # experiment configs
    parser.add_argument(
        "--exp-name",
        type=str,
        default="01_inference_prosody_annotation",
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )

    # dataset configs
    parser.add_argument(
        "--datasetpath",
        type=str,
        default="/mnt/audio_clip/webdataset_tar",
        help="The path to the dataset",
    )
    parser.add_argument(
        "--valid-datasets",
        nargs='+',
        type=str,
        default=None,
        help="Path to validation datasets",
    )
    parser.add_argument(
        "--featurelevel",
        type=str,
        choices=["wordlevel", "wordlevel-sorted", "wordlevel-sortedrandom", "wordlevel-sortedrandom-sortbyalph",
                 "wordboundarylevel", "wordboundarylevel-sortedrandom-sortbyalph",
                 "bigramlevel", "bigramlevel-sortedrandom-sortbyalph",
                 "sentlevel"],
        default="wordlevel",
        help="What level of features we want to use for training.",
    )
    parser.add_argument(
        "--class-label-path",
        type=str,
        default="",
        help="path to finetuning task labels",
    )
    parser.add_argument(
        "--dataset-proportion",
        type=float,
        default=1.0,
        help="How much proportion of dataset we want to train.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="The prefetch factor for dataloader. Larger value will use more memory and CPU but faster.",
    )

    # processing config
    parser.add_argument(
        "--data-filling",
        type=str,
        default="pad",
        help="type of data filling when the length is shorter than the max length."
             "Can be one of the following: repeat, repeatpad, pad",
    )
    parser.add_argument(
        "--data-truncating",
        type=str,
        default="cent_trunc",
        help="type of data truncation when the length is longer than the max length."
             "Can be one of the following: cent_trunc, rand_trunc, fusion",
    )

    # text process config
    parser.add_argument(
        "--text-augment-selection",
        type=str,
        default=None,
        help="For selecting levels of augmented text. Type is among ['all', 'augment_only', 'none']",
    )

    # saving setup
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--output-predictions",
        action="store_true",
        default=False,
        help="Output prediction results in validation.",
    )
    # validation setup
    parser.add_argument(
        "--no-eval",
        default=False,
        action="store_true",
        help="Training without evaluation.",
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=1,
        help="How often to run evaluation with val data.",
    )

    # model config
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="HTSAT-base-bert-base-uncased",
        help="Name of the model backbone to use.",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action="store_true",
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )

    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )

    parser.add_argument(
        "--split-opt",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )

    # training - warmup/finetune strategy
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="the path to the base checkpoint for finetuning",
    )
    parser.add_argument(
        "--lp-pretrained",
        type=str,
        default=None,
        help="the path to the finetuned checkpoint for another stage of finetuning",
    )
    parser.add_argument(
        "--freeze-text",
        default=False,
        action="store_true",
        help="if you need to freeze the text encoder, make this True",
    )
    parser.add_argument(
        "--freeze-text-after",
        type=int,
        default=-1,
        help="if you need to freeze the text encoder after (include) epoch x, "
             "set this param to x. Set -1 to disable it",
    )

    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action="store_true",
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )

    # arguments for distributed training

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged.",
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action="store_true",
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Default random seed.",
    )
    parser.add_argument(
        "--parallel-eval",
        default=False,
        action="store_true",
        help="Eval in parallel (multi-GPU, multi-node).",
    )

    # linear probe
    parser.add_argument(
        "--classification",
        default="bilstm",
        type=str,
        help="Options are ['bilstm']",
    )
    parser.add_argument(
        "--lp-mlp",
        default=False,
        action="store_true",
        help="Linear Probe using MLP layer or not.",
    )
    parser.add_argument(
        "--lp-text-freeze",
        default=False,
        action="store_true",
        help="Linear Probe using Freeze CLAP text encoder or not",
    )
    parser.add_argument(
        "--lp-audio-freeze",
        default=False,
        action="store_true",
        help="Linear Probe using Freeze CLAP audio encoder or not",
    )
    parser.add_argument(
        "--lp-audio-weight",
        type=float,
        default=1.,
        help="the weight of audio latent features to text latent features"
    )
    parser.add_argument(
        "--lp-act",
        default="softmax",
        type=str,
        help="Options are ['relu','elu','prelu','softmax','sigmoid']",
    )
    parser.add_argument(
        "--lp-loss", type=str, default="bce", help="Loss func of Linear Probe."
    )
    parser.add_argument(
        "--lp-celoss-weight",
        nargs='+',
        type=int,
        default=None,
        help="Weights of cross entropy loss",
    )
    parser.add_argument(
        "--lp-hier-labels",
        default=False,
        action="store_true",
        help="Use seperate classification FCs for seperate hierarchical labels",
    )
    parser.add_argument(
        "--lp-claploss",
        default=False,
        action="store_true",
        help="Add CLAP loss to downstream task loss or not",
    )
    parser.add_argument(
        "--lp-metrics",
        type=str,
        default="map,mauc,acc",
        help="Metrics of Linear Probe.",
    )
    parser.add_argument(
        "--lp-lr", type=float, default=1e-4, help="learning rate of linear probe"
    )
    # loss setup
    parser.add_argument(
        "--kappa", type=float, default=0,
        help="the kappa in the weighted contrastive loss, default is to turn off the weighted contrastive loss"
    )
    parser.add_argument(
        "--clap-mlploss",
        default=False,
        action="store_true",
        help="Using MLP loss for CLAP model or not",
    )

    parser.add_argument(
        "--wandb-id",
        type=str,
        default=None,
        help="the id of wandb experiment to restore.",
    )

    parser.add_argument(
        "--sleep", type=float, default=0, help="sleep n seconds before start training"
    )

    # variable length processing
    parser.add_argument(
        "--enable-fusion",
        default=False,
        action="store_true",
        help="Enable feature funsion for variable-length data",
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default='None',
        help="Type is among ['channel_map', 'daf_1d','aff_1d','iaff_1d','daf_2d','aff_2d','iaff_2d']",
    )

    args = parser.parse_args()

    return args
