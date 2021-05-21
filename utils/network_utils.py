import sys
sys.path.insert(0,'/home/maryam/Documents/Paper/HF')

from vgg import (vgg16_bn, 
                vgg19_bn, 
                vgg16, 
                vgg13)

#from mobilenetv2 import mobilenetv2
from mobilenetv2Impl2 import mobilenetv2


def get_network(network, **kwargs):
    networks = {
        'vgg16_bn': vgg16_bn,
        'vgg19_bn': vgg19_bn,
        'vgg16': vgg16,
        'vgg13': vgg13,
        'mobilenetv2': mobilenetv2
    }

    if network in ['vgg16', 'vgg19_bn', 'vgg13']:
        return networks[network](**kwargs).get_sequential_version()
    elif network in ['mobilenetv2']:
        return networks[network](**kwargs).get_sequential_version()

