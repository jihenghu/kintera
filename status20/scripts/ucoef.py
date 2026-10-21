import torch


def ucoef(th, dev="cpu"):
    """U = sum c_i (a_i + b_i T), c in mol/m^3, from kintera's VT->U state function (V = mass
    density = c / inv_mu); returns per-mole a, b and the max relative nonlinearity."""
    n = th.buffer("inv_mu").numel()
    E = torch.eye(n, dtype=torch.float64, device=dev)
    u1 = th.compute("VT->U", [E, torch.full((n,), 100., device=dev)])
    u2 = th.compute("VT->U", [E, torch.full((n,), 200., device=dev)])
    u3 = th.compute("VT->U", [E, torch.full((n,), 150., device=dev)])
    b = (u2 - u1) / 100.
    a = u1 - 100. * b
    nonlin = ((u3 - (a + 150. * b)).abs() / u3.abs().clamp_min(1e-300)).max().item()
    inv_mu = th.buffer("inv_mu")
    return (a / inv_mu).cpu().tolist(), (b / inv_mu).cpu().tolist(), nonlin
