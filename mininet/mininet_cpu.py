from .mininet import *


# MiniNet-v2-cpu model

class MiniNetv2CPU(MiniNetv2):
    """
    MiniNet-v2-cpu network. It is a smaller version of the MiniNet-v2 architecture
    suitable for training and inference on CPU. It removes the convolutional modules
    m3-m10, m16-m25 and m28-m29 from the original architecture.

    Note: Input tensors should have shape (B, in_channels, W, H) with W and H multiples of 8
    """
    def __init__(self, in_channels, out_channels, interpolate=True):
        super().__init__(in_channels, out_channels, interpolate)

        # 1. Downsample block (m1-m2)
        self.m_downsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(2)])

        # 2. Feature extractor block (m10-m15)
        rates = [1, 2, 1, 4, 1, 8]
        self.m_feature = nn.ModuleList([MiniNetv2Module(128, 128, rate) for rate in rates])

        # 4. Upsample block (m26-m27)
        self.m_upsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(2)])
