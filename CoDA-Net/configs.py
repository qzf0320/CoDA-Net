import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--gradient_clip', type=float, default=1.)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--T_0', type=int, default=20)
    parser.add_argument('--T_mult', type=int, default=2)
    parser.add_argument('--eta_min', type=float, default=1e-7)

    return parser.parse_args()
