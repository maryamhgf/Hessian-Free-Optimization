import sys
sys.path.insert(0,'/home/maryam/Documents/Paper/HF')

from vgg import (vgg16_bn, 
                vgg19_bn, 
                vgg16, 
                vgg13)

def get_network(network, **kwargs):
    networks = {
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'vgg16': vgg16,
        'vgg13': vgg13,
    }

    return networks[network](**kwargs)

