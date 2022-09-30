""" data """
import json
import os
import argparse
import numpy as np
import librosa
from mindspore import context

# config
parser = argparse.ArgumentParser(
    "Time-domain Audio Separation Network (TasNet) with Permutation Invariant "
    "Training")

# General config
# Task related
parser.add_argument('--test_dir', type=str, default="/tr",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--out_dir', type=str, default="/tr",
                    help='directory including mix.json, s1.json and s2.json')

parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
# Network architecture
parser.add_argument('--L', default=40, type=int,
                    help='Segment length (40=5ms at 8kHZ)')
parser.add_argument('--N', default=500, type=int,
                    help='The number of basis signals')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='Number of LSTM hidden units')
parser.add_argument('--num_layers', default=4, type=int,
                    help='Number of LSTM layers')
parser.add_argument('--bidirectional', default=0, type=int,
                    help='Whether use bidirectional LSTM')
parser.add_argument('--nspk', default=2, type=int,
                    help='Number of speaker')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', '-b', default=4, type=int,
                    help='Batch size')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer')
parser.add_argument('--lr', default=3e-4, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.01, type=float,
                    help='weight decay (L2 penalty)')


class DatasetGenerator:
    """ data """
    def __init__(self, json_dir, batch_size, sample_rate=8000, L=int(8000*0.005)):
        """
        Args:
            json_dir: directory including mix.json, s1.json and s2.json

        xxx_infos is a list and each item is a tuple (wav_file, #samples)
        """
        super(DatasetGenerator, self).__init__()
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')
        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)
        with open(s1_json, 'r') as f:
            s1_infos = json.load(f)
        with open(s2_json, 'r') as f:
            s2_infos = json.load(f)
        # sort it by #samples (impl bucket)
        def sort(infos):
            return sorted(infos, key=lambda info: (int(info[1]), info[0]), reverse=True)
        sorted_mix_infos = sort(mix_infos)
        sorted_s1_infos = sort(s1_infos)
        sorted_s2_infos = sort(s2_infos)

        mixture_pad = []
        lens = []
        source_pad = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start + batch_size)
            meta = [sorted_mix_infos[start:end], sorted_s1_infos[start:end], sorted_s2_infos[start:end], sample_rate, L]
            mixtures_pad, ilens, sources_pad = self.sort_and_pad(meta)
            for i in range(len(mixtures_pad)):
                mixture_pad.append(mixtures_pad[i])
                lens.append(ilens[i])
                source_pad.append(sources_pad[i])
            if end == len(sorted_mix_infos):
                break
            start = end

        self.mixture = mixture_pad
        self.len = lens
        self.sources = source_pad

    def __getitem__(self, index):
        return self.mixture[index], self.len[index], self.sources[index]

    def __len__(self):
        return len(self.mixture)

    def sort_and_pad(self, batch):
        # assert len(batch) == 1
        mixtures, sources = load_mixtures_and_sources(batch)

        # get batch of lengths of input sequences
        ilens = np.array([mix.shape[0] for mix in mixtures])

        # perform padding and convert to tensor
        mixtures_pad = pad_list([mix for mix in mixtures])
        sources_pad = pad_list([s for s in sources])
        # N x K x L x C -> N x C x K x L


        sources_pad = sources_pad.transpose((0, 3, 1, 2))
        return mixtures_pad, ilens, sources_pad

def load_mixtures_and_sources(batch):
    """
    Returns:
        mixtures: a list containing B items, each item is K x L np.ndarray
        sources: a list containing B items, each item is K x L x C np.ndarray
        K varies from item to item.
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, L = batch
    # for each utterance
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]
        s1_path = s1_info[0]
        s2_path = s2_info[0]
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]
        # read wav file
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)
        # Generate inputs and targets
        # K = int(np.ceil(len(mix) / L))
        # padding a little. mix_len + K > pad_len >= mix_len
        pad_len = 132800
        pad_mix = np.concatenate([mix, np.zeros([pad_len - len(mix)], np.float32)])
        pad_s1 = np.concatenate([s1, np.zeros([pad_len - len(s1)], np.float32)])
        pad_s2 = np.concatenate([s2, np.zeros([pad_len - len(s2)], np.float32)])
        # reshape
        mix = np.reshape(pad_mix, [3320, L])
        s1 = np.reshape(pad_s1, [3320, L])
        s2 = np.reshape(pad_s2, [3320, L])
        # merge s1 and s2
        s = np.dstack((s1, s2))  # K x L x C, C = 2
        # s = np.transpose(s, (2, 0, 1))  # C x K x L

        mixtures.append(mix)
        sources.append(s)
    return mixtures, sources

def pad_list(xs):
    n_batch = len(xs)

    max_len = max(x.shape for x in xs)
    if len(max_len) == 2:
        pad = np.zeros((n_batch, max_len[0], max_len[1]), np.float32)
    else:
        pad = np.zeros((n_batch, max_len[0], max_len[1], max_len[2]), np.float32)

    for i in range(n_batch):
        temp = xs[i].shape
        pad[i, :temp[0]] = xs[i]
    return pad

if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=1)
    args = parser.parse_args()
    print(args)
    tr_dataset = DatasetGenerator(args.test_dir, args.batch_size,
                                  sample_rate=args.sample_rate, L=args.L)
    output_path = args.out_dir

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=False)
    cnt = 1
    for data in tr_dataset:
        mixture, _, _ = data
        savename = os.path.join(output_path + str(cnt) + '.bin')
        mixture.tofile(savename)
        cnt += 1
