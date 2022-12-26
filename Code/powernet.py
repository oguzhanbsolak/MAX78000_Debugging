###################################################################################################
# PowerNet network
# Victor Luder
###################################################################################################
"""
PowerNet network description
"""
from signal import pause
import torch
from torch import nn

import ai8x

import matplotlib
import matplotlib.pyplot as plt

"""
Network description class
"""
class PowerNet(nn.Module):
    
    
    def __init__(self, num_classes=None, dimensions=(1, 103), num_channels=1, bias=False, **kwargs):
        super().__init__()
        torch.set_default_dtype(torch.float64)

        # assert dimensions[0] == dimensions[1]  # Only square supported

        # Keep track of image dimensions so one constructor works for all image sizes
        dim_x, dim_y = dimensions

        self.fc1 = ai8x.FusedLinearReLU(103, 64, bias=True, **kwargs)

        self.fc2 = ai8x.FusedLinearReLU(64, 144, bias=True, **kwargs)
        
        self.fc3 = ai8x.FusedLinearReLU(144, 36, bias=True, **kwargs)

        self.fc4 = ai8x.Linear(36, 1, wide=True, bias=True, **kwargs)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        """

    """
    Assemble the model
    """
    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Data plotting - for debug
        # matplotlib.use('MacOSX')
        # plt.imshow(x[0, 0], cmap="gray")
        # plt.show()
        # breakpoint()
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x


def powernet(pretrained=False, **kwargs):

    assert not pretrained
    return PowerNet(**kwargs)

"""
Network description
"""
models = [
    {
        'name': 'powernet',
        'min_input': 1,
        'dim': 1,
    }
]

