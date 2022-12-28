import argparse
import torch

# 1. 创建命令解析器
parser = argparse.ArgumentParser()

# 2. 添加参数
parser.add_argument('--train-file', default='data/train.txt')
parser.add_argument('--dev-file', default='data/dev.txt')

parser.add_argument('--UNK', default=0, type=int)
parser.add_argument('--PAD', default=1, type=int)

# TODO 常改动参数
parser.add_argument('--type', default='train', help='train or evaluate')
parser.add_argument('--gpu', default=4, type=int, help='master gpu')
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--n_layers', default=2, type=int, help='layers of transformers')
parser.add_argument('--n_heads', default=8, type=int, help='heads of MultiHeadAttention')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--d-ff', default=1024, type=int)
parser.add_argument('--dropout', default=0.1, type=float)

parser.add_argument('--max-length', default=60, type=int)

parser.add_argument('--save-file', default='save/model.pt')
parser.add_argument('--vocab-file', default='vocab')

parser.add_argument('--src', default='en')
parser.add_argument('--tgt', default='zh')

# 3. 解析对象parser
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
args.device = device
