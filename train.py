""" Train """
import os
import argparse
import json
import librosa
from data import DatasetGenerator
from Loss import Loss
from tasnet import TasNet
from network_define import WithLossCell
from train_wrapper import TrainingWrapper
import mindspore.dataset as ds
from mindspore import context
from mindspore import nn
from mindspore.train import Model
from mindspore.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from mindspore import load_checkpoint, load_param_into_net
from mindspore import set_seed
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode

set_seed(1)
parser = argparse.ArgumentParser("Time-domain Audio Separation Network (TasNet) with Permutation Invariant "
                                 "Training")
parser.add_argument('--in-dir', type=str, default=r"/home/work/user-job-dir/inputs/data/",
                    help='Directory path of wsj0 including tr, cv and tt')
parser.add_argument('--out-dir', type=str, default=r"/home/work/user-job-dir/inputs/data_json",
                    help='Directory path to put output files')
parser.add_argument('--sample-rate', type=int, default=8000,
                    help='Sample rate of audio file')
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default='./data')
parser.add_argument('--train_url',
                    help='model folder to save/load',
                    default='./model')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'ModelArts'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--device_id', type=int, default=0,
                    help='device id')
parser.add_argument('--continue_train', type=int,
                    default=0,
                    help='Whether to continue training')
parser.add_argument('--model_path', type=str,
                    default="",
                    help='Path to model file created by training')
# Task related
parser.add_argument('--train_dir', type=str, default=r"/tr",
                    help='data directory including mix.json, s1.json and s2.json')
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
# Training config
parser.add_argument('--epochs', default=50, type=int,
                    help='Number of maximum epochs')
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
# save and load model
parser.add_argument('--save_folder', default="./ckpt",
                    help='Location to save epoch models')

def preprocess_one_dir(in_dir, out_dir, out_filename, sample_rate=8000):
    """
    sample_rate: 8000
    Read the wav file and save the path and len to the json file
    """
    file_infos = []
    in_dir = os.path.abspath(in_dir)
    wav_list = os.listdir(in_dir)
    for wav_file in wav_list:
        if not wav_file.endswith('.wav'):
            continue
        wav_path = os.path.join(in_dir, wav_file)
        samples, _ = librosa.load(wav_path, sr=sample_rate)
        file_infos.append((wav_path, len(samples)))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, out_filename + '.json'), 'w') as f:
        json.dump(file_infos, f, indent=4)

def preprocess(arg):
    """ Process all files """
    print("Begin preprocess")
    for data_type in ['tr']:
        for speaker in ['mix', 's1', 's2']:
            preprocess_one_dir(os.path.join(arg.in_dir, data_type, speaker),
                               os.path.join(arg.out_dir, data_type),
                               speaker,
                               sample_rate=arg.sample_rate)
    print("Preprocess done")

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    if args.device_target == 'Ascend':
        context.set_context(device_id=args.device_id)
    if args.device_target == 'ModelArts':
        import moxing as mox
        obs_data_url = args.data_url
        args.data_url = '/home/work/user-job-dir/inputs/data/'
        obs_train_url = args.train_url
        args.train_url = '/home/work/user-job-dir/outputs/model/'
        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url,
                                                      args.data_url))

    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        is_distributed = 'False'
    elif device_num > 1:
        is_distributed = 'True'

    if is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        rank_id = get_rank()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)
        context.set_auto_parallel_context(parameter_broadcast=True)
        args.save_folder = os.path.join(args.save_folder, 'ckpt_' + str(get_rank()) + '/')
        print("Starting traning on multiple devices.")

    if args.device_target == 'ModelArts':
        args.save_folder = args.train_url
        preprocess(args)

    print("Preparing Data")
    tr_dataset = DatasetGenerator(args.train_dir, args.batch_size,
                                  sample_rate=args.sample_rate, L=args.L)
    if is_distributed == 'True':
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"], shuffle=True,
                                        num_shards=rank_size, shard_id=rank_id)
    else:
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"], shuffle=True)
    tr_loader = tr_loader.batch(batch_size=args.batch_size)
    print("Prepare Data done")

    # model
    net = TasNet(args.L, args.N, args.hidden_size, args.num_layers,
                 bidirectional=bool(args.bidirectional), nspk=args.nspk)
    if args.continue_train == 1:
        home = os.path.dirname(os.path.realpath(__file__))
        ckpt = os.path.join(home, args.model_path)
        print('=====> load params into generator')
        params = load_checkpoint(ckpt)
        load_param_into_net(net, params)
        print('=====> finish load generator')

    print(net)
    num_steps = tr_loader.get_dataset_size()

    milestone = [10 * num_steps, 40 * num_steps, 50 * num_steps]
    learning_rates = [1e-3, 3e-4, 1e-4]
    lr = nn.piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.Adam(net.get_parameters(), learning_rate=lr, weight_decay=args.l2, loss_scale=0.01)
    my_loss = Loss()
    loss_cb = LossMonitor()
    time_cb = TimeMonitor()
    net_with_loss = WithLossCell(net, my_loss)
    net_with_clip_norm = TrainingWrapper(net_with_loss, optimizer)
    net_with_clip_norm.set_train()

    config_ck = CheckpointConfig(save_checkpoint_steps=num_steps, keep_checkpoint_max=1)
    if args.device_target == 'ModelArts':
        ckpt_cb = ModelCheckpoint(prefix='TasNet_train',
                                  directory=args.save_folder + '/device_' + os.getenv('DEVICE_ID') + '/',
                                  config=config_ck)
    else:
        ckpt_cb = ModelCheckpoint(prefix='TasNet_train',
                                  directory=args.save_folder,
                                  config=config_ck)
    cb = [time_cb, loss_cb, ckpt_cb]
    model = Model(net_with_clip_norm)

    print("Training......", flush=True)
    model.train(epoch=args.epochs, train_dataset=tr_loader, callbacks=cb, dataset_sink_mode=True)

    if args.device_target == 'ModelArts':
        mox.file.copy_parallel(args.train_url, obs_train_url)
        print("Successfully Upload {} to {}".format(args.train_url,
                                                    obs_train_url))
