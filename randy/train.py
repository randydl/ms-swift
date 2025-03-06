import sys
from omegaconf import OmegaConf
from llamafactory.train.tuner import run_exp


def parse_args():
    results = []
    for arg in sys.argv[1:]:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            content = OmegaConf.load(arg)
            for key, value in content.items():
                results.append(f'--{key}={value}')
        else:
            results.append(arg)
    return results


if __name__ == '__main__':
    args = parse_args()
    sys.argv = [sys.argv[0]] + args
    run_exp()
