import numpy as np
import pytest

try:
    import torch
except ImportError:                      # torch not installed / unusable -> skip these tests
    pytest.skip("torch not available", allow_module_level=True)
from src.train_autoencoder import AutoEncoder, recon_error, set_seed  # noqa: E402


def test_autoencoder_shapes_and_bottleneck():
    m = AutoEncoder(n_in=20, hidden=[16], latent=4)
    x = torch.randn(7, 20)
    assert m.encoder(x).shape == (7, 4) and m(x).shape == (7, 20)


def test_recon_error_is_per_row_and_nonnegative():
    set_seed(0)
    m = AutoEncoder(10, [8], 3)
    X = np.random.default_rng(0).normal(size=(50, 10)).astype(np.float32)
    e = recon_error(m, X)
    assert e.shape == (50,) and (e >= 0).all()


def test_same_seed_gives_identical_weights():
    set_seed(1); a = AutoEncoder(10, [8], 3)
    set_seed(1); b = AutoEncoder(10, [8], 3)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_state_dict_roundtrip(tmp_path):
    set_seed(2); a = AutoEncoder(10, [8], 3)
    torch.save(a.state_dict(), tmp_path / "w.pt")
    b = AutoEncoder(10, [8], 3); b.load_state_dict(torch.load(tmp_path / "w.pt"))
    X = np.random.default_rng(1).normal(size=(5, 10)).astype(np.float32)
    np.testing.assert_allclose(recon_error(a, X), recon_error(b, X), rtol=1e-6)
