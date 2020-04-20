import torch.nn as nn
import torch.nn.functional as F
from torchmeta.modules import MetaModule
from collections import OrderedDict

class MetaConvTranspose2d(nn.ConvTranspose2d,MetaModule):
    __doc__ = nn.ConvTranspose2d.__doc__

    def forward(self, input, params=None,output_size=None):
        # type: (Tensor, Optional[List[int]]) -> Tensor

        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        return F.conv_transpose2d(
            input,params['weight'], bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)