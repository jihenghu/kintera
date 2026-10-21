"""K19 review sweep over tests/jupiter.yaml.
usage: python -I sweep.py <pyinstall> <yaml> <cpu|cuda> <out.pt>
Two 5000-state families, generated on the host with torch.Generator seed 0 / 1,
so every device sees bit-identical inputs:
  wide: T ~ U[100, 350] K, P ~ logU[1e4, 1e7] Pa, dry 1, each of H2O/NH3/H2S zero
        w.p. 1/4 else logU[1e-5, 5e-2], each of the 5 clouds zero w.p. 1/2 else
        logU[1e-8, 1e-2]; normalized.
  cold: same, but T ~ U[100, 110] K and NH3 gas zero w.p. 1/2.
  plus the PR's test cell (UV 1955 of Linfeng's sweep) appended as the last state.
Modes: TP and UV(kkt) at max_iter 100 and 1000, UV(auto) at 100.
"""
import sys
sys.path.insert(0, sys.argv[1])
import torch, kintera
from kintera import ThermoOptions, ThermoY, ThermoX

torch.set_default_dtype(torch.float64)
yaml, dev, out = sys.argv[2], torch.device(sys.argv[3]), sys.argv[4]
N, NS = 5000, 9
CELL = ([0.9452025167266452, 3.0793244963643104e-06, 0.0, 0.04432008639206904, 0.0,
         0.0033088825716865882, 0.007165383153972112, 0.0, 5.1831130667826045e-08],
        104.18617226219762, 1946753.4681013762)


def lu(g, n, lo, hi):
    return torch.exp(torch.empty(n).uniform_(torch.log(torch.tensor(lo)).item(),
                                             torch.log(torch.tensor(hi)).item(), generator=g))


def states(seed, tlo, thi, p_nh3_zero):
    g = torch.Generator().manual_seed(seed)
    T = torch.empty(N).uniform_(tlo, thi, generator=g)
    P = lu(g, N, 1e4, 1e7)
    X = torch.zeros(N, NS)
    X[:, 0] = 1.
    for i in (1, 2, 3):
        pz = p_nh3_zero if i == 2 else 0.25
        X[:, i] = torch.where(torch.rand(N, generator=g) < pz, 0., lu(g, N, 1e-5, 5e-2))
    for i in range(4, 9):
        X[:, i] = torch.where(torch.rand(N, generator=g) < 0.5, 0., lu(g, N, 1e-8, 1e-2))
    X /= X.sum(-1, keepdim=True)
    X = torch.cat([X, torch.tensor([CELL[0]])])
    T = torch.cat([T, torch.tensor([CELL[1]])])
    P = torch.cat([P, torch.tensor([CELL[2]])])
    return T, P, X


res = {"kintera": kintera.__file__}
for fam, args in {"wide": (0, 100., 350., 0.25), "cold": (1, 100., 110., 0.5)}.items():
    T, P, X = (t.to(dev) for t in states(*args))
    res[fam + "/in"] = (T.cpu(), P.cpu(), X.cpu())
    n = T.numel()
    for mi in (100, 1000):
        op = ThermoOptions.from_yaml(yaml).max_iter(mi)
        tx = ThermoX(op); tx.to(dev)
        x = X.clone(); d = torch.zeros(n, 1, device=dev)
        tx.forward(T, P, x, False, d)
        res[f"{fam}/TP{mi}"] = (d.cpu().squeeze(-1), T.cpu(), x.cpu())
    for solver, mis in (("kkt", (100, 1000)), ("auto", (100,))):
        for mi in mis:
            op = ThermoOptions.from_yaml(yaml).max_iter(mi).uv_solver(solver)
            th = ThermoY(op); th.to(dev)
            inv_mu = th.buffer("inv_mu")
            c = X * (P / (8.31446 * T) / X[:, :4].sum(-1))[:, None]
            rho = (c / inv_mu).sum(-1)
            y = (c[:, 1:] / inv_mu[1:] / rho[:, None]).T.contiguous()
            V = th.compute("DY->V", [rho, y])
            U = th.compute("VT->U", [V, T])
            d = torch.zeros(n, 1, device=dev)
            th.forward(rho, U, y, False, d)
            V = th.compute("DY->V", [rho, y])
            Tf = th.compute("VU->T", [V, U])
            res[f"{fam}/UV{solver}{mi}"] = (d.cpu().squeeze(-1), Tf.cpu(), (V * inv_mu).cpu())
            if mi == 100 and solver == "kkt":
                res[f"{fam}/UVin"] = (rho.cpu(), U.cpu(), c.cpu())
torch.save(res, out)
for k, v in res.items():
    if k.endswith("in") or k == "kintera":
        continue
    d, mi = v[0], int(k.split("TP")[-1].split("kkt")[-1].split("auto")[-1])
    fires = ((d - torch.trunc(d)).abs() * 1000).round()
    d = torch.trunc(d)
    st20 = d == -(2000 + mi)
    other = (d < 0) & ~st20
    print(k, "fail", int((d < 0).sum()), "st20", int(st20.sum()), "other", int(other.sum()),
          sorted(set(int(-z) for z in d[other].tolist()))[:12], "fired_cells", int((fires > 0).sum()))
print(res["kintera"])
