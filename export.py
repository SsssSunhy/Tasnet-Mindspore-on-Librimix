import argparse
import numpy as np
from tasnet import TasNet
import mindspore
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

parser = argparse.ArgumentParser()
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
parser.add_argument('--B', default=1, type=int,
                    help='batch size')
parser.add_argument('--K', default=3320, type=int,
                    help='Max length divide L')
parser.add_argument('--ckpt_path', type=str, default="/ckpt",
                    help='Checkpoint path')

def export_TasNet():
    args = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)

    net = TasNet(args.L, args.N, args.hidden_size, args.num_layers,
                 bidirectional=bool(args.bidirectional), nspk=args.nspk)
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param_dict)

    input_mixture = Tensor(np.ones([args.B, args.K, args.L]), mindspore.float32)
    export(net, input_mixture, file_name='TasNet_MindIR', file_format='MINDIR')

if __name__ == '__main__':
    export_TasNet()
