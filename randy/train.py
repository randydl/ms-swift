import sys
import importlib
from omegaconf import OmegaConf


def parse_args():
    mode, args = '', []

    for arg in sys.argv[1:]:
        if not arg.startswith('-') and arg.endswith('.yaml'):
            conf = OmegaConf.load(arg)
            path = conf.pop('dataset_info')
            info = OmegaConf.load(path)
            mode = conf.pop('stage')

            for k, v in conf.items():
                if k == 'dataset':
                    v = v.split(',')
                    for i, d in enumerate(v):
                        name, *rest = d.split('#', 1)
                        name = info[name]['file_name']
                        v[i] = '#'.join([name] + rest)
                    args.append(f'--{k}')
                    args.extend(v)
                else:
                    args.append(f'--{k}={v}')
        else:
            args.append(arg)

    return mode, args


if __name__ == '__main__':
    mode, args = parse_args()
    sys.argv = [sys.argv[0]] + args
    module = importlib.import_module('swift.llm')
    getattr(module, f'{mode}_main')()
