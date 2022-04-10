#!/usr/bin/env python
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

prog_description = """\
Used to compare the initialization method between models
"""

def get_state_dict(model, key=None):
    if key is not None:
        state_dict = model[key]
    elif "state_dict" in model:
        state_dict = model['state_dict']
    elif "model" in model:
        state_dict = model['model']
    else:
        state_dict = model
    assert isinstance(state_dict, dict)
    return state_dict


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=prog_description)
    parser.add_argument('model_a', type=Path)
    parser.add_argument('model_b', type=Path)
    parser.add_argument('--draw',
                        action='store_true',
                        help='draw the KDE of variables')
    parser.add_argument('-p',
                        default=0.05,
                        type=float,
                        help='the threshold of p-value')
    args = parser.parse_args()
    return args

#  def compare_distribution(state_dict_a, state_dict_b, p_thres):
#      for k, v1 in state_dict_a.items():
#          if k in state_dict_b:
#              v2 = state_dict_b[k]
#              v1 = v1.cpu().flatten()
#              v2 = v2.cpu().flatten()
#              pvalue = abs(v1 - v2).sum()
#              if pvalue < p_thres:
#                  yield k, pvalue, v1, v2
#          else:
#              print(f'Key "{k}" not found')

def compare_distribution(state_dict_a, state_dict_b, p_thres):
    assert len(state_dict_a) == len(state_dict_b)
    for k, v1 in state_dict_a.items():
        assert k in state_dict_b
        v2 = state_dict_b[k]
        v1 = v1.cpu().flatten()
        v2 = v2.cpu().flatten()
        pvalue = stats.kstest(v1, v2).pvalue
        if pvalue < p_thres:
            yield k, pvalue, v1, v2


def main():
    args = parse_args()
    state_dict_a = get_state_dict(torch.load(args.model_a))
    state_dict_b = get_state_dict(torch.load(args.model_b))
    for key, p, v1, v2 in compare_distribution(state_dict_a, state_dict_b, args.p):
        print(' -------------------- ')
        print(f"\033[92m'key-{len(v1)}'\033[0m: {key}")
        print(f"\033[92m'p'\033[0m: {p}")
        if args.draw:
            perm_ids = torch.randperm(v1.size(0))[:30000]
            print(v1[perm_ids])
            sns.kdeplot(v1[perm_ids], shade=True, label=args.model_a.stem)
            sns.kdeplot(v2[perm_ids], shade=True, label=args.model_b.stem)
            plt.legend()
            plt.title(key)
            plt.show()


if __name__ == "__main__":
    main()
