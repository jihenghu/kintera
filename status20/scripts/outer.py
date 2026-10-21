"""Outer brute-force equilibrium for selected cells vs the solver's accepted state.
usage: outer.py <yaml> <pyinstall> head.pt inst.pt base.pt out.pt"""
import sys
sys.path.insert(0, sys.argv[2]); sys.path.insert(0, "<SCRIPTDIR>")
import torch, mpmath as mp
from multiprocessing import Pool
import brute
from ucoef import ucoef
torch.set_default_dtype(torch.float64)
H, I, B = torch.load(sys.argv[3]), torch.load(sys.argv[4]), torch.load(sys.argv[5])


def status(d, mi):
    d = torch.trunc(d)
    return 0 if d >= 0 else (20 if d == -(2000 + mi) else int(-d) // 100)


def work(job):
    mp.mp.dps = 50
    m, i, kind, inp, a, b = job
    if kind == "TP":
        T, P, X = inp
        sols = brute.eq_tp(X, T, P)
    else:
        T0, U, c = inp
        sols = brute.eq_uv(c, T0, a, b, U)
    adm = {k: v for k, v in sols.items() if isinstance(v, dict) and v["ok"]}
    return m, i, adm, {k: (v if not isinstance(v, dict) else None) for k, v in sols.items() if not isinstance(v, dict)}


if __name__ == "__main__":
    from kintera import ThermoOptions, ThermoY
    th = ThermoY(ThermoOptions.from_yaml(sys.argv[1]))
    a, b, nl = ucoef(th)
    jobs = []
    for m in ["wide/TP100", "wide/UVkkt100", "cold/TP100", "cold/UVkkt100"]:
        fam, mode = m.split("/")
        d = I[m][0]
        fired = (((d - torch.trunc(d)).abs() * 1000).round() > 0)
        sh = torch.tensor([status(v, 100) for v in H[m][0]])
        sb = torch.tensor([status(v, 100) for v in B[m][0]])
        sel = (fired | (sh != sb)).nonzero().squeeze(-1).tolist()
        for i in sel:
            if mode.startswith("TP"):
                T, P, X = (t[i].tolist() for t in B[fam + "/in"])
                jobs.append((m, i, "TP", (T, P, X), a, b))
            else:
                rho, U, c = B[fam + "/UVin"]
                jobs.append((m, i, "UV", (B[fam + "/in"][0][i].item(), U[i].item(), c[i].tolist()), a, b))
    with Pool(28) as p:
        R = p.map(work, jobs, chunksize=1)
    out = []
    for (m, i, adm, bad), job in zip(R, jobs):
        fam = m.split("/")[0]
        d = I[m][0][i]
        rec = dict(mode=m, cell=i, fired=int(((d - torch.trunc(d)).abs() * 1000).round()),
                   head=status(H[m][0][i], 100), base=status(B[m][0][i], 100), nadm=len(adm), unsolved=list(bad))
        if len(adm) >= 1:
            ref = min(adm.values(), key=lambda v: 0)  # report all below
            rec["assemblages"] = list(adm)
            for tag, S in (("head", H), ("base", B)):
                if rec[tag] != 0:
                    continue
                diffs = []
                for k, v in adm.items():
                    if m.endswith("TP100"):
                        got = S[m][2][i].tolist(); want = v["x"]; dT = 0.
                    else:
                        got = S[m][2][i].tolist(); want = v["c"]; dT = abs(float(v["T"]) - S[m][1][i].item())
                    ab = [abs(float(w) - g) for w, g in zip(want, got)]
                    rl = [abs(float(w) - g) / abs(float(w)) for w, g in zip(want, got) if abs(float(w)) > 1e-30]
                    rl0 = [abs(g) for w, g in zip(want, got) if abs(float(w)) <= 1e-30]
                    diffs.append((max(ab), max(rl), max(rl0 + [0.]), dT, k))
                rec[tag + "_diff"] = min(diffs)
            rec["refT"] = [float(v["T"]) for v in adm.values()]
        out.append(rec)
    torch.save(out, sys.argv[6])
    for r in out:
        print(r)
