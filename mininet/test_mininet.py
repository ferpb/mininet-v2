import torch
from . import MiniNetv2, MiniNetv2CPU

def test_mininet_output_size_same_as_input():
    input = torch.rand((1, 3, 1024, 512))
    model = MiniNetv2(3, 1)
    output = model(input)
    assert input.shape[2:] == output.shape[2:]

def test_mininet_cpu_output_size_same_as_input():
    input = torch.rand((1, 3, 1024, 512))
    model = MiniNetv2CPU(3, 1)
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
