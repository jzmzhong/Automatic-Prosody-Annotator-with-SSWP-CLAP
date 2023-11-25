import logging
import os
from contextlib import suppress

import numpy as np
import torch


def evaluate(model, data, args, labels=[]):
    """Evaluate the model on the given data.

    Args:
        model (clap_module.linear_probe_words2sent.LinearProbe): The model that has finished loading
            the pretrained weights
        data (dict): Waiting for the inferred data
        args (argparse.Namespace): Command-line arguments
        labels (list, optional): Label mapping from ../../class_labels/prosodic_boundaries_labels_nopunc.json
            {"LW": 0, "PW": 1, "PPH": 2, "IPH": 3}. Defaults to [].
    """

    device = torch.device(args.device)
    model.eval()

    if args.output_predictions:
        out_results_file = open(os.path.join(args.logs, args.exp_name, "out_results.txt"), encoding="utf-8", mode="w")
    else:
        out_results_file = None

    autocast = torch.cuda.amp.autocast if args.precision == "amp" else suppress
    if "valid" in data and args.val_frequency:
        if args.parallel_eval:
            dataloader, sampler = data["valid"].dataloader, data["valid"].sampler
            samples_per_val = data["valid"].num_samples
        else:
            dataloader = data["valid"].dataloader
            num_samples = 0
            samples_per_val = data["valid"].num_samples

        eval_info = {
            'pred': [],
        }
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            for i, batch in enumerate(dataloader):
                audios = batch['audio']
                texts = batch['text']

                with autocast():
                    pred, mask = model(texts, audios, device=device)

                    if out_results_file:
                        for sent_data, sent_tar, sent_tar_mask, sent_pred, sent_pred_mask in zip(batch['text'],
                                                                                                 batch['boundary'],
                                                                                                 batch['boundary_mask'],
                                                                                                 pred, mask):
                            sent_id = sent_data["sent_id"]
                            sent_pred = torch.masked_select(sent_pred, ~sent_pred_mask.repeat(1, len(labels)))
                            sent_pred = torch.reshape(sent_pred, (-1, len(labels)))
                            sent_pred = torch.argmax(sent_pred, -1).cpu()
                            sent_tar = torch.masked_select(sent_tar, ~sent_tar_mask)
                            out_results_file.write(sent_id + "\n")
                            out_results_file.write(np.array2string(sent_pred.numpy(), max_line_width=99999) + "\n")
                            out_results_file.write(np.array2string(sent_tar.numpy(), max_line_width=99999) + "\n")

                    if not args.lp_hier_labels:
                        pred = torch.masked_select(pred, ~mask.repeat(1, 1, len(labels)))
                        pred = torch.reshape(pred, (-1, len(labels)))
                        eval_info['pred'].append(pred)
                    else:
                        pred_l3 = (torch.masked_select(pred[:, :, 0:1], ~mask) >= 0.5).int()
                        pred_l4 = (torch.masked_select(pred[:, :, 1:2], ~mask) >= 0.5).int()
                        pred_l5 = (torch.masked_select(pred[:, :, 2:], ~mask) >= 0.5).int()
                        pred_l2 = ((pred_l3 + pred_l4 + pred_l5) == 0).int()
                        pred_l3 = ((pred_l3 - pred_l4 - pred_l5) > 0).int()
                        pred_l4 = ((pred_l4 - pred_l5) > 0).int()
                        pred_l2345 = torch.vstack([pred_l2, pred_l3, pred_l4, pred_l5]).transpose(0, 1)
                        eval_info['pred'].append(pred_l2345)

            eval_info['pred'] = torch.cat(eval_info['pred'], dim=0).cpu()

            if out_results_file:
                out_results_file.close()
