import pytest
import torch

from ldctbench.hub import Methods, load_model


@pytest.fixture(scope="session")
def sample_input():
    x = torch.randn(1, 1, 32, 32)  # make this small to speed up testing
    yield x
    del x


@torch.no_grad()
def test_apply_cnn10(sample_input):
    net = load_model(Methods.CNN10)
    y = net(sample_input)
    assert y.shape == sample_input.shape


@torch.no_grad()
def test_load_redcnn(sample_input):
    net = load_model(Methods.REDCNN)
    y = net(sample_input)
    assert y.shape == sample_input.shape


@torch.no_grad()
def test_load_wganvgg(sample_input):
    net = load_model(Methods.WGANVGG)
    y = net(sample_input)
    assert y.shape == sample_input.shape


@torch.no_grad()
def test_load_resnet(sample_input):
    net = load_model(Methods.RESNET)
    y = net(sample_input)
    assert y.shape == sample_input.shape


@torch.no_grad()
def test_load_qae(sample_input):
    net = load_model(Methods.QAE)
    y = net(sample_input)
    assert y.shape == sample_input.shape


@torch.no_grad()
def test_load_dugan(sample_input):
    net = load_model(Methods.DUGAN)
    y = net(sample_input)
    assert y.shape == sample_input.shape


@torch.no_grad()
def test_load_transct(sample_input):
    x = torch.randn(1, 1, 512, 512)  # required input size for TransCT
    net = load_model(Methods.TRANSCT)
    y = net(x)
    assert y.shape == x.shape


# Do not test Bilateral here as it needs a separate GPU-only python package installed
