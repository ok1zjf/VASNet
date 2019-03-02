__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"


import os
import numpy as np
import subprocess
import platform
import sys
import pkg_resources
import torch
import h5py
import json
import ortools
from torch.nn.modules.module import _addindent
# import PIL as Image
# import cv2

def list_files(path, extensions=[], sort=True, max_len=-1):
    if os.path.isdir(path):
        filenames = [os.path.join(path, fn) for fn in os.listdir(path) if
                           any([fn.lower().endswith(ext) for ext in extensions])]
    else:
        print("ERROR. ", path,' is not a directory!')
        return []

    if sort:
        filenames.sort()

    if max_len>-1:
        filenames = filenames[:max_len]

    return filenames


def del_file(filename):
    try:
        os.remove(filename)
    except:
        pass
    return

def get_video_list(video_path, max_len=-1, extensions=['avi', 'flv', 'mpg', 'mp4']):
    return list_files(video_path, extensions=extensions , sort=True, max_len=max_len)

def get_image_list(video_path, max_len=-1):
    return list_files(video_path, extensions=['jpg', 'jpeg', 'png'], sort=True, max_len=max_len)

def run_command(command):
    p = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return '\n'.join([ '\t'+line.decode("utf-8").strip() for line in p.stdout.readlines()])

def ge_pkg_versions():
    dep_versions = {}
    dep_versions['display'] = run_command('cat /proc/driver/nvidia/version')

    dep_versions['cuda'] = 'NA'
    cuda_home = '/usr/local/cuda/'
    if 'CUDA_HOME' in os.environ:
        cuda_home = os.environ['CUDA_HOME']

    cmd = cuda_home+'/version.txt'
    if os.path.isfile(cmd):
        dep_versions['cuda'] = run_command('cat '+cmd)

    dep_versions['cudnn'] = torch.backends.cudnn.version()
    dep_versions['platform'] = platform.platform()
    dep_versions['python'] = sys.version_info[:3]
    dep_versions['torch'] = torch.__version__
    dep_versions['numpy'] = np.__version__
    dep_versions['h5py'] = h5py.__version__
    dep_versions['json'] = json.__version__
    dep_versions['ortools'] = ortools.__version__
    dep_versions['torchvision'] = pkg_resources.get_distribution("torchvision").version

    # dep_versions['PIL'] = Image.VERSION
    # dep_versions['OpenCV'] = 'NA'
    # if 'cv2' in sys.modules:
    #     dep_versions['OpenCV'] = cv2.__version__


    return dep_versions


def print_pkg_versions():
    print("Packages & system versions:")
    print("----------------------------------------------------------------------")
    versions = ge_pkg_versions()
    for key, val in versions.items():
        print(key,": ",val)
    print("")
    return


def torch_summarize(model, show_weights=True, show_parameters=True):
    """Summarizes torch model by showing trainable parameters and weights."""
    tmpstr = model.__class__.__name__ + ' (\n'
    parameters = 0
    convs = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [torch.nn.modules.container.Container, torch.nn.modules.container.Sequential]:
            modstr, p, cnvs = torch_summarize(module)
            parameters += p
            convs += cnvs
        else:
            modstr = module.__repr__()
            convs += len(modstr.split('Conv2d')) - 1

        modstr = _addindent(modstr, 2)
        # if 'conv' in key:
        #     convs += 1

        params = sum([np.prod(p.size()) for p in module.parameters()])
        parameters += params
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={} / {}'.format(params, parameters)
        tmpstr += ', convs={}'.format(convs)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr, parameters, convs


def print_table(table, cell_width=[3,35,8]):
    slen=sum(cell_width)+len(cell_width)*2+2
    print('-'*slen)
    header = table.pop(0)
    for i, head in enumerate(header):
        print('  {name: <{alignment}}'.format(name=head, alignment=cell_width[i]), end='')

    print('')
    print('='*slen)
    for row in table:
        for i, val in enumerate(row):
            print('  {val: <{alignment}}'.format(val=val, alignment=cell_width[i]), end='')
        print('')
    print('-'*slen)


if __name__ == "__main__":
    # Tests
    print_pkg_versions()
