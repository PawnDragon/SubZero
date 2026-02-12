import torch
from torch import nn
import torch.nn.functional as F


def _random_orthonormal(shape, device, dtype):
    q, _ = torch.linalg.qr(torch.randn(shape, device=device, dtype=dtype))
    return q


class SubMuonLinear(nn.Module):
    """
    Linear layer with frozen base weight and trainable low-rank core X in a
    fixed subspace (U, V). Effective weight: W + U @ X @ V.
    """

    def __init__(self, linear: nn.Linear, rank: int):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("SubMuonLinear expects nn.Linear input")
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        w = linear.weight.detach()
        self.weight = nn.Parameter(w.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        r = int(rank)
        device = self.weight.device
        dtype = self.weight.dtype
        U = _random_orthonormal((self.out_features, r), device, dtype)
        V = _random_orthonormal((self.in_features, r), device, dtype).T.contiguous()
        self.register_buffer("U", U.contiguous())
        self.register_buffer("V", V.contiguous())
        self.X = nn.Parameter(torch.zeros((r, r), device=device, dtype=dtype))

    def forward(self, x):
        delta = self.U @ self.X @ self.V
        weight_eff = self.weight + delta
        return F.linear(x, weight_eff, self.bias)


def inject_submuon_linear(model: nn.Module, rank: int):
    """
    Replace all nn.Linear modules with SubMuonLinear.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, SubMuonLinear(module, rank))
        else:
            inject_submuon_linear(module, rank)
    return model
