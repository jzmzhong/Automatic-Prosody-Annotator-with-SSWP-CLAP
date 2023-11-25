import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time

from clap_module import create_model
from evaluate.data_words2sent import get_data
from evaluate.params import parse_args
from evaluate.logger import setup_logging
from evaluate.lp_evaluate_words2sent import evaluate
from clap_module.utils import load_class_label
from clap_module.linear_probe_words2sent import LinearProbe


def main():
    args = parse_args()

    time.sleep(args.sleep)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    args.class_index_dict = load_class_label(args.class_label_path)

    args.log_path = None

    log_base_path = os.path.join(args.logs, args.exp_name)
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = f"out-{args.rank}" if args.log_local else "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)

    # avoid log dir in same name:
    postfix = 0
    while os.path.exists(args.log_path):
        postfix += 1
        log_base_path_new = log_base_path + '-' + str(postfix)
        os.makedirs(log_base_path_new, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path_new, log_filename)

    # Set logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # fully initialize device environment
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    args.device = device
    device = torch.device(device)

    # Create CLAP model
    # model
    clap_model, clap_model_cfg = create_model(
        args,
        args.model_name,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
    )

    args.lp_out_ch = len(list(args.class_index_dict.keys()))
    # Linear Probe
    logging.info(f"linear probe using mlp: {args.lp_mlp}")
    logging.info(f"linear probe using text freeze: {args.lp_text_freeze}")
    logging.info(f"linear probe using audio freeze: {args.lp_audio_freeze}")
    logging.info(f"linear probe act layer: {args.lp_act}")
    logging.info(f"linear probe in ch: {clap_model.joint_embed_shape}")
    logging.info(f"linear probe out ch: {args.lp_out_ch}")
    logging.info(f"linear probe audio to text weight: {args.lp_audio_weight}")

    model = LinearProbe(
        clap_model,
        text_freeze=args.lp_text_freeze,
        audio_freeze=args.lp_audio_freeze,
        in_ch=clap_model.joint_embed_shape,
        out_ch=args.lp_out_ch,
        audio_weight=args.lp_audio_weight,
        classification=args.classification,
        hier_labels=args.lp_hier_labels,
        act=args.lp_act
    )
    print(model)
    if args.lp_pretrained:
        pretrained_model = torch.load(args.lp_pretrained, map_location='cpu')["state_dict"]
        if next(iter(pretrained_model.items()))[0].startswith("module"):
            pretrained_model = {k[len("module."):]: v for k, v in pretrained_model.items()}
        model.load_state_dict(pretrained_model, strict=False)
        logging.info(f"Loading finetuned CLAP model weights !!!")
    model = model.to(device)

    logging.info("Linear Probe CLAP Model:")
    logging.info(f"{str(clap_model)}")
    logging.info("Params:")
    params_file = os.path.join(args.logs, args.exp_name, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")

    data = get_data(args, clap_model_cfg)

    assert len(data), "At least one train or eval dataset must be specified."

    # optionally resume from a checkpoint
    cudnn.benchmark = True
    cudnn.deterministic = False

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != "none"

    evaluate(model, data, args, labels=args.class_index_dict)


if __name__ == "__main__":
    main()
