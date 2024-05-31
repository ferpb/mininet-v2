import torch
from .mininet import MiniNetV2Segmentation, MiniNetV2SegmentationCPU

def test_mininet_output_size_same_as_input():
    input = torch.rand((1, 3, 1024, 512))
    model = MiniNetV2Segmentation(3, 1, interpolate=True)
    output = model(input)
    assert input.shape[2:] == output.shape[2:]

def test_mininet_cpu_output_size_same_as_input():
    input = torch.rand((1, 3, 1024, 512))
    model = MiniNetV2SegmentationCPU(3, 1, interpolate=True)
    output = model(input)
    assert input.shape[2:] == output.shape[2:]

def test_mininet_num_parameters():
    model = MiniNetV2Segmentation(3, 12)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert pytorch_total_params == 513034

def test_mininet_cpu_num_parameters():
    model = MiniNetV2SegmentationCPU(3, 12)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert pytorch_total_params == 269194

def test_mininet_layers():
    model = MiniNetV2Segmentation(3, 10, interpolate=False)
    input = torch.randn((1, 3, 512, 1024), dtype=torch.float32)

    # Downsample
    d1 = model.encoder.downsample_1(input)
    assert d1.shape == (1, 16, 256, 512)
    d2 = model.encoder.downsample_2(d1)
    assert d2.shape == (1, 64, 128, 256)

    assert len(model.encoder.downsample_modules) == 10
    m_out = d2
    for m in model.encoder.downsample_modules:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    d3 = model.encoder.downsample_3(m_out)
    assert d3.shape == (1, 128, 64, 128)

    # Feature extractor
    assert len(model.encoder.feature_modules) == 16
    m_out = d3
    for m in model.encoder.feature_modules:
        m_out = m(m_out)
        assert m_out.shape == (1, 128, 64, 128)

    # Refinement
    d4 = model.aux_downsample_4(input)
    assert d4.shape == (1, 16, 256, 512)

    d5 = model.aux_downsample_5(d4)
    assert d5.shape == (1, 64, 128, 256)

    # Upsample
    up1 = model.upsample_1(m_out)
    assert up1.shape == (1, 64, 128, 256)

    assert len(model.upsample_mods) == 4
    m_out = up1 + d5
    for m in model.upsample_mods:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    out = model.output_conv(m_out)
    assert out.shape == (1, 10, 256, 512)

def test_mininet_cpu_layers():
    model = MiniNetV2SegmentationCPU(3, 10, interpolate=False)
    input = torch.randn((1, 3, 512, 1024), dtype=torch.float32)

    # Downsample
    d1 = model.encoder.downsample_1(input)
    assert d1.shape == (1, 16, 256, 512)
    d2 = model.encoder.downsample_2(d1)
    assert d2.shape == (1, 64, 128, 256)

    assert len(model.encoder.downsample_modules) == 2
    m_out = d2
    for m in model.encoder.downsample_modules:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    d3 = model.encoder.downsample_3(m_out)
    assert d3.shape == (1, 128, 64, 128)

    # Feature extractor
    assert len(model.encoder.feature_modules) == 6
    m_out = d3
    for m in model.encoder.feature_modules:
        m_out = m(m_out)
        assert m_out.shape == (1, 128, 64, 128)

    # Refinement
    d4 = model.aux_downsample_4(input)
    assert d4.shape == (1, 16, 256, 512)

    d5 = model.aux_downsample_5(d4)
    assert d5.shape == (1, 64, 128, 256)

    # Upsample
    up1 = model.upsample_1(m_out)
    assert up1.shape == (1, 64, 128, 256)

    assert len(model.upsample_mods) == 2
    m_out = up1 + d5
    for m in model.upsample_mods:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    out = model.output_conv(m_out)
    assert out.shape == (1, 10, 256, 512)
