import subprocess
import sys
import os

CUDA = 'cu102'
TORCH = '1.9.0'


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


if '/home/program' in os.path.join(os.path.dirname(os.path.abspath(__file__))):
    # KP environment
    install('pandas')
    install('tqdm')
    subprocess.call([sys.executable, "-m", "pip", "install",
                     'torch-scatter', 'torch-sparse', 'torch-cluster', 'torch-spline-conv', 'torch-geometric',
                     '-f', 'https://pytorch-geometric.com/whl/torch-{}+{}'.format(TORCH, CUDA)])
