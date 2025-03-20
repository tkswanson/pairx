import torch
from zennit.canonizers import SequentialMergeBatchNorm, AttributeCanonizer, CompositeCanonizer
from zennit.layer import Sum
from timm.models.resnet import Bottleneck as ResNetBottleneck
from timm.models.resnet import BasicBlock as ResNetBasicBlock


# based on https://github.com/chr5tphr/zennit/blob/master/src/zennit/torchvision.py, but for timm-based ResNet
class ResNetBottleneckCanonizerTimm(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of timm.models.resnet.ResNet type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        #x = torch.stack([shortcut, x], dim=-1)
        #x = self.canonizer_sum(x)
        x += shortcut

        self.act3.inplace=False
        x = self.act3(x)

        return x

class ResNetBasicBlockCanonizerTimm(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out


class ResNetCanonizerTimm(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ResNetBottleneckCanonizerTimm(),
            ResNetBasicBlockCanonizerTimm(),
        ))


