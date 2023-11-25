import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from clap_module.utils import load_class_label


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    num_batches: -1
    num_samples: -1


class CLAPDataset_LinearProbe(Dataset):
    """linear probe dataset for CLAP model
    """
    def __init__(self, datasetpath, featurelevel, datasetname,
                 pretrained_tokenizer, is_wav=False, audio_config=None):
        """init function for CLAPDataset_LinearProbe

        Args:
            datasetpath (str): The root directory of the dataset
            featurelevel (str): ["wordboundarylevel", "bigramlevel"]. Defaults to "wordboundarylevel".
            datasetname (str): dataset name
            pretrained_tokenizer (torch.nn.Module): pretrained tokenizer from transformers
            is_wav (bool, optional): if True, amodel.startswith("hubert"). Defaults to False.
            audio_config (dict, optional): audio config. Defaults to None.
        """
        # extract all data sample paths
        ling_dir = os.path.join(datasetpath, "{}_ling".format(featurelevel.split("-")[0]), datasetname)
        print(ling_dir)
        ling_paths = [os.path.join(ling_dir, ling_file) for ling_file in sorted(os.listdir(ling_dir))]

        if is_wav:
            context_time = audio_config["context_time_bins"] * audio_config["frame_rate"]
            max_time = audio_config["max_time_bins"] * audio_config["frame_rate"]
            self.ling_paths = []
            audio_dir = os.path.join(datasetpath, "sentlevel_wav", datasetname)
            all_audio_paths = [os.path.join(audio_dir, audio_file) for audio_file in sorted(os.listdir(audio_dir))]
            self.audio_paths = []
            for ling_file, ling_path in zip(sorted(os.listdir(ling_dir)), ling_paths):
                audio_file = "-".join(ling_file.split(".")[0].split("-")[:-1]) + ".wav.npy"
                audio_path = os.path.join(audio_dir, audio_file)
                if audio_path in all_audio_paths:
                    f_read = open(ling_path, encoding="utf-8", mode="r")
                    token_ling = json.load(f_read)
                    f_read.close()
                    if context_time and token_ling["end"] - token_ling["start"] >= context_time * 2:
                        print("Skipping Ling Due to Lengthy Word/Punc:", ling_path)
                    elif max_time and token_ling["end"] - token_ling["start"] >= max_time * 2:
                        print("Skipping Ling Due to Lengthy Word/Punc:", ling_path)
                    else:
                        self.ling_paths.append(ling_path)
                        self.audio_paths.append(audio_path)
                else:
                    print("Skipping Wav Due to Missing File:", audio_path)
        else:
            self.ling_paths = ling_paths
            audio_dir = os.path.join(datasetpath, "{}_mel".format(featurelevel.split("-")[0]), datasetname)
            self.audio_paths = [os.path.join(audio_dir, audio_file) for audio_file in sorted(os.listdir(audio_dir))]
            assert len(self.ling_paths) == len(self.audio_paths), "Unequal number of samples: {}, {}".format(
                len(self.ling_paths), len(self.mel_paths))

        # group ling paths and audio paths based on sentence id
        self.sent_ling_paths = []
        self.sent_audio_paths = []
        prev_sent_id = None
        for ling_path, audio_path in zip(self.ling_paths, self.audio_paths):
            # sent_id = ling_path.split("/")[-1].split("-")[0]
            sent_id = "-".join(ling_path.split("/")[-1].split("-")[:-1])
            if sent_id == prev_sent_id:
                self.sent_ling_paths[-1].append(ling_path)
                self.sent_audio_paths[-1].append(audio_path)
            else:
                self.sent_ling_paths.append([ling_path])
                self.sent_audio_paths.append([audio_path])
            prev_sent_id = sent_id

        print("Number of samples loaded:", self.__len__())

        self.pretrained_tokenizer = pretrained_tokenizer
        self.is_wav = is_wav

        # feature configs
        if featurelevel.split("-")[0] in ["wordboundarylevel", "bigramlevel"]:
            self.multi_words = True
        else:
            self.multi_words = False

    def __len__(self):
        return len(self.sent_ling_paths)

    def __getitem__(self, idx):
        ling_paths = self.sent_ling_paths[idx]
        audio_paths = self.sent_audio_paths[idx]

        samples = []

        for ling_path, audio_path in zip(ling_paths, audio_paths):

            # sent_id = ling_path.split("/")[-1].split("-")[0]
            sent_id = "-".join(ling_path.split("/")[-1].split("-")[:-1])

            f_read = open(ling_path, encoding="utf-8", mode="r")
            token_ling = json.load(f_read)
            f_read.close()

            if self.is_wav:
                sent_wav = np.load(audio_path)
                assert "-".join(ling_path.split("/")[-1].split("-")[:-1]) + ".wav.npy" == audio_path.split("/")[
                    -1], "path mismatch"
            else:
                token_mel = np.load(audio_path)
                assert ling_path.split("/")[-1].replace(".json", ".mel.npy") == audio_path.split("/")[
                    -1], "path mismatch"

            # text related info & target label
            sample = {"idx": idx,
                      "token_text": token_ling["word"].strip("\"\'-"),
                      "token_index": token_ling["word_index"],
                      "full_text": token_ling["full_sent"].strip(),
                      "boundary": None}
            if "punc_index" in token_ling and self.multi_words:
                sample["punc_index"] = token_ling["punc_index"]
                if sample["punc_index"]:
                    tokenized_result, word_start, word_end = self.tokenizer_multi(sample["full_text"],
                                                                                  sample["token_index"],
                                                                                  sample["punc_index"],
                                                                                  sample["token_text"],
                                                                                  self.pretrained_tokenizer)
                else:
                    tokenized_result, word_start, word_end = self.tokenizer(sample["full_text"], sample["token_index"],
                                                                            sample["token_text"],
                                                                            self.pretrained_tokenizer)
            elif "bigram_index" in token_ling and self.multi_words:
                sample["bigram_index"] = token_ling["bigram_index"]
                if sample["bigram_index"]:
                    tokenized_result, word_start, word_end = self.tokenizer_multi(sample["full_text"],
                                                                                  sample["token_index"],
                                                                                  sample["bigram_index"],
                                                                                  sample["token_text"],
                                                                                  self.pretrained_tokenizer)
                else:
                    raise Exception("Data Error! {token_ling}.")
            else:
                tokenized_result, word_start, word_end = self.tokenizer(sample["full_text"], sample["token_index"],
                                                                        sample["token_text"], self.pretrained_tokenizer)
            sample["tokenized_result"] = tokenized_result
            sample["word_start"] = word_start
            sample["word_end"] = word_end

            # audio related info
            if self.is_wav:
                sample["sent_wav"] = sent_wav
                sample["audio_start"] = token_ling["start"]
                sample["audio_end"] = token_ling["end"]
            else:
                sample["token_mel"] = token_mel

            # label2index
            if boundary_labels:
                # process boundary, only in finetuning
                boundary = str(token_ling["boundary"])
                if boundary in boundary_labels:
                    boundary = boundary_labels[boundary]
                else:
                    print("Warning! word: {}, boundary: {}, sent: {}".format(token_ling["word"], boundary,
                                                                             token_ling["full_sent"]))
                    boundary = 0
                sample["boundary"] = boundary

            # add sentence id
            sample["sent_id"] = sent_id
            samples.append(sample)

        return samples

    def tokenizer(self, text, word_index, word, pretrained_tokenizer, max_length=77, lower=True):
        # print(text, word_index, word)
        if lower:
            text = text.lower()
            word = word.lower()
        result = pretrained_tokenizer(text.strip().split(), \
                                      padding="max_length", truncation=True, max_length=77, stride=10, \
                                      return_overflowing_tokens=True, return_tensors="pt", \
                                      is_split_into_words=True)
        batch_index = 0

        # locate the batch the word and its corresponding subwords are in
        word_start_end = result.word_to_tokens(batch_index, word_index)
        while word_start_end is None:
            batch_index += 1
            word_start_end = result.word_to_tokens(batch_index, word_index)

        # postprocess a special case: a word with subwords spanning from the first batch to the second
        BPE_tokens = pretrained_tokenizer.convert_ids_to_tokens(result["input_ids"][batch_index])
        word_start, word_end = word_start_end
        BPE_word = "".join([_.strip("#") for _ in BPE_tokens[word_start: word_end]])
        if BPE_word != word and word.startswith(BPE_word):
            print(BPE_word, word, text)
            batch_index += 1
            # update word_start, word_end, BPE_tokens, and BPE_word
            word_start_end = result.word_to_tokens(batch_index, word_index)
            word_start, word_end = word_start_end
            BPE_tokens = pretrained_tokenizer.convert_ids_to_tokens(result["input_ids"][batch_index])
            BPE_word = "".join([_.strip("#") for _ in BPE_tokens[word_start: word_end]])

        # avoid misalignment between words in data and words in tokenized result
        # strict data check
        assert word_start_end is not None, (word, word_index, text)
        assert BPE_word.replace("-", "").replace("\"", "").replace("\'", "") == \
               word.replace("-", "").replace("\"","").replace(
            "\'", ""), (word, BPE_word, BPE_tokens[word_start: word_end], word_index, word_start, text, BPE_tokens)

        sent_encoded = {
            "input_ids": result["input_ids"][batch_index],
            "token_type_ids": result["token_type_ids"][batch_index],
            "attention_mask": result["attention_mask"][batch_index],
        }

        return sent_encoded, word_start, word_end

    def tokenizer_multi(self, text, word_index_start, word_index_end, word, pretrained_tokenizer, max_length=77,
                        lower=True):
        if lower:
            text = text.lower()
            word = word.lower()
        result = pretrained_tokenizer(text.strip().split(), \
                                      padding="max_length", truncation=True, max_length=77, stride=10, \
                                      return_overflowing_tokens=True, return_tensors="pt", \
                                      is_split_into_words=True)
        batch_index = 0

        # locate the batch the word and its corresponding subwords are in
        word_start_subwords = result.word_to_tokens(batch_index, word_index_start)
        word_end_subwords = result.word_to_tokens(batch_index, word_index_end)
        max_iter = 1000  # 1000 batches of 77 subwords as max input
        while (word_start_subwords is None or word_end_subwords is None) and max_iter > 0:
            batch_index += 1
            word_start_subwords = result.word_to_tokens(batch_index, word_index_start)
            word_end_subwords = result.word_to_tokens(batch_index, word_index_end)
            max_iter -= 1
        if max_iter <= 0:
            raise Exception("Data Error! {text}, {word_index_start}, {word_index_end}, {word}.")

        # postprocess a special case: a word with subwords spanning from the first batch to the second
        BPE_tokens = pretrained_tokenizer.convert_ids_to_tokens(result["input_ids"][batch_index])
        word_start, word_end = word_start_subwords[0], word_end_subwords[-1]
        BPE_word = "".join([_.strip("#") for _ in BPE_tokens[word_start: word_end]])
        if BPE_word.replace("-", "") != word.replace("-", "").replace("#", "") and\
                word.replace("-", "").replace("#", "").startswith(BPE_word.replace("-", "")):
            batch_index += 1
            # update word_start, word_end, BPE_tokens, and BPE_word
            word_start_subwords = result.word_to_tokens(batch_index, word_index_start)
            word_end_subwords = result.word_to_tokens(batch_index, word_index_end)
            word_start, word_end = word_start_subwords[0], word_end_subwords[-1]
            BPE_tokens = pretrained_tokenizer.convert_ids_to_tokens(result["input_ids"][batch_index])
            BPE_word = "".join([_.strip("#") for _ in BPE_tokens[word_start: word_end]])

        # avoid misalignment between words in data and words in tokenized result
        # strict data check
        assert word_start_subwords is not None, (word, word_index_start, word_index_end, text)
        assert word_end_subwords is not None, (word, word_index_start, word_index_end, text)
        assert BPE_word.replace("-", "") == word.replace("-", "").replace("#", ""), (
        word, BPE_tokens[word_start: word_end], word_index_start, word_index_end, word_start, word_end, text,
        BPE_tokens)

        sent_encoded = {
            "input_ids": result["input_ids"][batch_index],
            "token_type_ids": result["token_type_ids"][batch_index],
            "attention_mask": result["attention_mask"][batch_index],
        }

        return sent_encoded, word_start, word_end


def CLAPCollate_LinearProbe(batch, data_filling, data_truncating, max_mel_len=100, is_wav=False, audio_config=None,
                            hier_labels=False):
    """Collate function for CLAP model

    Args:
        batch (list): _description_
        data_filling (str): The method of populating the data
        data_truncating (str): the methods of data truncation
        max_mel_len (int, optional): The maximum length of the MEL input. Defaults to 100.
        is_wav (bool, optional): if True, amodel.startswith("hubert"). Defaults to False.
        audio_config (dict, optional): audio config. Defaults to None.
        hier_labels (bool, optional): for word with punc/sil only. Defaults to False.

    Returns:
        sent_features (dict): the features of the sentence
    """
    sent_features = {"text": [],
                     "audio": []}
    if boundary_labels:
        sent_features["boundary"] = []

    for samples in batch:

        token_indices = torch.tensor([[sample["word_start"], sample["word_end"]] for sample in samples],
                                     dtype=torch.int)

        sent_input_ids = torch.stack([sample["tokenized_result"]["input_ids"].long() for sample in samples])
        sent_token_type_ids = torch.stack([sample["tokenized_result"]["token_type_ids"].long() for sample in samples])
        sent_attention_mask = torch.stack([sample["tokenized_result"]["attention_mask"].long() for sample in samples])

        text_features = {
            "token_indices": token_indices,
            "input_ids": sent_input_ids,
            "token_type_ids": sent_token_type_ids,
            "attention_mask": sent_attention_mask,
            "sent_id": samples[0]["sent_id"]
        }

        if is_wav:
            if audio_config["context_time_bins"]:
                sent_wavs = []
                token_indices = []
                for sample in samples:
                    start_index, end_index = int(sample["audio_start"] * audio_config["sampling_rate"]), int(
                        sample["audio_end"] * audio_config["sampling_rate"])
                    context_indices = int(
                        audio_config["context_time_bins"] * audio_config["frame_rate"] * audio_config["sampling_rate"])
                    if end_index - start_index > context_indices * 2:
                        raise Exception("Data Error:", sample)
                    else:
                        cent_index = int((start_index + end_index) / 2)
                        start_index_2 = max(cent_index - context_indices, 0)
                        end_index_2 = min(cent_index + context_indices, len(sample["sent_wav"]))
                        # print(sample["sent_wav"].shape, start_index_2, end_index_2)
                        sent_wavs.append(sample["sent_wav"][start_index_2: end_index_2])
                        token_indices.append(
                            torch.tensor([start_index - start_index_2, end_index - start_index_2], dtype=torch.float) /
                            audio_config["sampling_rate"])
                audio_features = {
                    "sent_wavs": sent_wavs,
                    "token_indices": token_indices
                }
            else:
                token_wavs = []
                for sample in samples:
                    start_index, end_index = int(sample["audio_start"] * audio_config["sampling_rate"]), int(
                        sample["audio_end"] * audio_config["sampling_rate"])
                    token_wavs.append(sample["sent_wav"][start_index: end_index])
                audio_features = {
                    "token_wavs": token_wavs,
                }
        else:
            mels = []
            for sample in samples:
                mel = torch.tensor(sample["token_mel"], dtype=torch.float).permute(1, 0)
                if mel.shape[0] > max_mel_len:
                    if data_truncating == "front_trunc":
                        mel = mel[:max_mel_len, :]
                    elif data_truncating == "back_trunc":
                        mel = mel[-max_mel_len:, :]
                    elif data_truncating == "cent_trunc":
                        mel = mel[int(0.5 * (mel.shape[0] - max_mel_len)): int(0.5 * (mel.shape[0] + max_mel_len)), :]
                    else:
                        raise NotImplementedError
                mels.append(mel)

            if data_filling == "pad":
                mels.append(torch.ones((max_mel_len, samples[0]["token_mel"].shape[0]), dtype=float))
                mels_mask = pad_sequence(mels, batch_first=True, padding_value=-999.)[:-1, :, :1] == -999.
                mels = pad_sequence(mels, batch_first=True, padding_value=0.)[:-1, :, :]
            else:
                raise NotImplementedError
            mels_mask = mels_mask.permute(0, 2, 1)

            assert mels.shape[1] == max_mel_len, mels.shape
            assert mels_mask.shape[2] == max_mel_len, mels_mask.shape

            audio_features = {
                "mel": mels,
                "mel_mask": mels_mask,
            }

        sent_features["text"].append(text_features)
        sent_features["audio"].append(audio_features)
        if boundary_labels:
            boundaries = torch.tensor([sample["boundary"] for sample in samples], dtype=torch.long)
            # boundaries[-1] = 3. # sentence end set to be L5
            # boundaries[boundaries == 3.] = 0. # combine L2 and L5
            sent_features["boundary"].append(boundaries)
    sent_features["boundary_mask"] = pad_sequence(sent_features["boundary"], batch_first=True, padding_value=-1)[:,
                                     :] == -1
    sent_features["boundary"] = pad_sequence(sent_features["boundary"], batch_first=True, padding_value=0)
    if hier_labels:  # for word with punc/sil only
        sent_features["boundary_L3"] = pad_sequence((sent_features["boundary"] >= 1).float(), batch_first=True,
                                                    padding_value=0)
        sent_features["boundary_L4"] = pad_sequence((sent_features["boundary"] >= 2).float(), batch_first=True,
                                                    padding_value=0)
        sent_features["boundary_L5"] = pad_sequence((sent_features["boundary"] >= 3).float(), batch_first=True,
                                                    padding_value=0)
    return sent_features


def get_data(args, model_cfg):
    """Load data via the DataLoader method

    Args:
        args (argparse.Namespace): Parameters such as data paths and label information
        model_cfg (dict): The type of model used for inference

    Returns:
        (dict): The data loader for inference
    """
    global boundary_labels
    if args.class_label_path:
        boundary_labels = load_class_label(args.class_label_path)
    else:
        print("No Class Labels Found! Ignore in pretraining mode!")
        boundary_labels = None

    # load pretrained tokenizer
    tmodel = model_cfg["text_cfg"]["model_type"] + "-" + model_cfg["text_cfg"]["model_name"]
    if tmodel.startswith("bert"):
        from transformers import BertTokenizerFast
        try:
            pretrained_tokenizer = BertTokenizerFast.from_pretrained(tmodel)
        except:
            pretrained_tokenizer = BertTokenizerFast.from_pretrained("prajjwal1/{}".format(tmodel))
    elif tmodel.startswith("roberta"):
        from transformers import RobertaTokenizerFast
        pretrained_tokenizer = RobertaTokenizerFast.from_pretrained(tmodel)
    elif tmodel.startswith("bart"):
        from transformers import BartTokenizerFast
        pretrained_tokenizer = BartTokenizerFast.from_pretrained("facebook/{}".format(tmodel))
    else:
        raise NotImplementedError

    # audio configs
    amodel = model_cfg["audio_cfg"]["model_type"] + "-" + model_cfg["audio_cfg"]["model_name"]
    if amodel.startswith("hubert"):
        is_wav = True
    else:
        is_wav = False
    audio_config = model_cfg["audio_cfg"]

    hier_labels = args.lp_hier_labels

    # load valid datasets
    valid_datasets_names = args.valid_datasets
    valid_datasets = []
    for valid_dataset_name in valid_datasets_names:
        valid_datasets.append(
            CLAPDataset_LinearProbe(args.datasetpath, args.featurelevel, valid_dataset_name, pretrained_tokenizer,
                                    is_wav, audio_config))
    valid_datasets = ConcatDataset(valid_datasets)

    # dataloader
    data_filling = args.data_filling
    data_truncating = args.data_truncating
    max_mel_len = int(model_cfg["audio_cfg"]["max_time_bins"])

    if args.featurelevel.startswith("wordlevel-sorted") or args.featurelevel.startswith(
            "wordboundarylevel-sorted") or args.featurelevel.startswith("bigramlevel-sorted"):
        raise NotImplementedError
    elif args.featurelevel in ["wordlevel", "wordboundarylevel", "bigramlevel"]:
        valid_dataloader = DataLoader(valid_datasets, batch_size=args.batch_size,
                                      collate_fn=lambda batch: CLAPCollate_LinearProbe(batch, data_filling,
                                                                                       data_truncating, max_mel_len,
                                                                                       is_wav, audio_config,
                                                                                       hier_labels),
                                      shuffle=True, drop_last=False,
                                      num_workers=args.workers, prefetch_factor=2)
    else:
        raise NotImplementedError

    # get num of batches and num of samples
    num_batches_valid = len(valid_dataloader)
    num_samples_valid = len(valid_datasets)

    return {
        "valid": DataInfo(valid_dataloader, None, num_batches_valid, num_samples_valid)
    }


if __name__ == "__main__":
    pass
