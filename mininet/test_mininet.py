import torch
from . import MiniNetv2, MiniNetv2CPU

def test_mininet_output_size_same_as_input():
    input = torch.rand((1, 3, 1024, 512))
    model = MiniNetv2(3, 1, interpolate=True)
    output = model(input)
    assert input.shape[2:] == output.shape[2:]

def test_mininet_cpu_output_size_same_as_input():
    input = torch.rand((1, 3, 1024, 512))
    model = MiniNetv2CPU(3, 1, interpolate=True)
    output = model(input)
    assert input.shape[2:] == output.shape[2:]

def test_mininet_num_parameters():
    model = MiniNetv2(3, 12)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert pytorch_total_params == 522892

def test_mininet_cpu_num_parameters():
    model = MiniNetv2CPU(3, 12)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert pytorch_total_params == 270092

def test_mininet_layers():
    model = MiniNetv2(3, 10, interpolate=False)
    input = torch.randn((1, 3, 512, 1024), dtype=torch.float32)

    # Downsample
    d1 = model.d1(input)
    assert d1.shape == (1, 16, 256, 512)
    d2 = model.d2(d1)
    assert d2.shape == (1, 64, 128, 256)

    assert len(model.m_downsample) == 10
    m_out = d2
    for m in model.m_downsample:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    d3 = model.d3(m_out)
    assert d3.shape == (1, 128, 64, 128)

    # Feature extractor
    assert len(model.m_feature) == 16
    m_out = d3
    for m in model.m_feature:
        m_out = m(m_out)
        assert m_out.shape == (1, 128, 64, 128)

    # Refinement
    d4 = model.d4(input)
    assert d4.shape == (1, 16, 256, 512)

    d5 = model.d5(d4)
    assert d5.shape == (1, 64, 128, 256)

    # Upsample
    up1 = model.up1(m_out)
    assert up1.shape == (1, 64, 128, 256)

    assert len(model.m_upsample) == 4
    m_out = up1 + d5
    for m in model.m_upsample:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    out = model.output(m_out)
    assert out.shape == (1, 10, 256, 512)

def test_mininet_cpu_layers():
    model = MiniNetv2CPU(3, 10, interpolate=False)
    input = torch.randn((1, 3, 512, 1024), dtype=torch.float32)

    # Downsample
    d1 = model.d1(input)
    assert d1.shape == (1, 16, 256, 512)
    d2 = model.d2(d1)
    assert d2.shape == (1, 64, 128, 256)

    assert len(model.m_downsample) == 2
    m_out = d2
    for m in model.m_downsample:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    d3 = model.d3(m_out)
    assert d3.shape == (1, 128, 64, 128)

    # Feature extractor
    assert len(model.m_feature) == 6
    m_out = d3
    for m in model.m_feature:
        m_out = m(m_out)
        assert m_out.shape == (1, 128, 64, 128)

    # Refinement
    d4 = model.d4(input)
    assert d4.shape == (1, 16, 256, 512)

    d5 = model.d5(d4)
    assert d5.shape == (1, 64, 128, 256)

    # Upsample
    up1 = model.up1(m_out)
    assert up1.shape == (1, 64, 128, 256)

    assert len(model.m_upsample) == 2
    m_out = up1 + d5
    for m in model.m_upsample:
        m_out = m(m_out)
        assert m_out.shape == (1, 64, 128, 256)

    out = model.output(m_out)
    assert out.shape == (1, 10, 256, 512)
