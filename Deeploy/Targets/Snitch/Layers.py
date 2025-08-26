import math
from typing import List

from Deeploy.DeeployTypes import NodeMapper, ONNXLayer


class ConvLayer(ONNXLayer):

    def __init__(self, maps: List[NodeMapper]):
        super().__init__(maps)

    def computeOps(self) -> int:
        operatorRepresentation = self.mapper.parser.operatorRepresentation

        width = operatorRepresentation['dim_im_out_x']
        height = operatorRepresentation.get('dim_im_out_y', 1)
        channel_in = operatorRepresentation['ch_im_in']
        channel_out = operatorRepresentation['ch_im_out']
        kernel_shape = operatorRepresentation['kernel_shape']
        group = operatorRepresentation.get('group', 1)

        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(channel_in, int)
        assert isinstance(channel_out, int)
        assert isinstance(kernel_shape, (list, tuple))
        assert isinstance(group, int)

        return width * height * math.prod(kernel_shape) * channel_in * channel_out // group * 2
