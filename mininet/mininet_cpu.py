from .mininet import *


# MiniNet-v2-cpu model

class MiniNetv2CPU(nn.Module):
    """
    MiniNet-v2-cpu network. It is a smaller version of the MiniNet-v2 architecture
    suitable for training and inference on CPU

    Note: Input tensors should have shape (B, in_channels, W, H) with W and H multiples of 8
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1. Downsample block
        self.d1 = MiniNetv2Downsample(in_channels, 16, depthwise=False)
        self.d2 = MiniNetv2Downsample(16, 64, depthwise=False)
        self.m_downsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(2)])
        self.d3 = MiniNetv2Downsample(64, 128, depthwise=False)

        # 2. Feature extractor block
        rates = [1, 2, 1, 4, 1, 8]
        self.m_feature = nn.ModuleList([MiniNetv2Module(128, 128, rate) for rate in rates])

        # 3. Refinement block
        self.d4 = MiniNetv2Downsample(in_channels, 16, depthwise=False)
        self.d5 = MiniNetv2Downsample(16, 64, depthwise=False)

        # 4. Upsample block
        self.up1 = MiniNetv2Upsample(128, 64)
        self.m_upsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(2)])
        self.output = MiniNetv2Upsample(64, out_channels)

    def forward(self, x):
        d1 = self.d1(x)

        d2 = self.d2(d1)
        m_downsample = d2
        for m in self.m_downsample:
            m_downsample = m(m_downsample) # m1-m2
        d3 = self.d3(m_downsample)

        m_feature = d3
        for m in self.m_feature:
            m_feature = m(m_feature) # m10-m15

        d4 = self.d4(x)
        d5 = self.d5(d4)

        up1 = self.up1(m_feature)

        m_upsample = up1 + d5
        for m in self.m_upsample:
            m_upsample = m(m_upsample) # m26-m27

        output = self.output(m_upsample)

        output = Interpolate(size=x.shape[2:])(output)
        output = F.softmax(output, dim=1)
        return output
