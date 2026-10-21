"""Independent brute-force references for the K19 review (no kintera solver involved).
qp(A,b,C,d): min ||A x - b||^2 s.t. C x <= d by enumerating active sets (mpmath, 300 digits).
eq_uv / eq_tp: tests/jupiter.yaml equilibrium by enumerating the 8 cloud assemblages of
H2O(l), NH3(s), NH4SH(s); svp formulas copied from src/vapors/vapor_functions.h, and the
caloric model U = sum c_i (a_i + b_i T) whose coefficients are read off kintera's "VT->U"
state function (linearity checked by the caller)."""
import itertools
import math
import mpmath as mp

mp.mp.dps = 300
R = mp.mpf("8.31446")
LN10 = mp.log(10)


def _ideal(T, tr, pr, bl, gl, bs, gs):
    t = T / tr
    b, g = (bl, gl) if T > tr else (bs, gs)
    return (1 - 1 / t) * b - g * mp.log(t) + mp.log(pr)


def lsvp(j, T):
    if j == 0:
        return _ideal(T, mp.mpf("273.16"), mp.mpf("611.7"), mp.mpf("24.845"), mp.mpf("4.986009"),
                      mp.mpf("22.98"), mp.mpf("0.52"))
    if j == 1:
        return _ideal(T, mp.mpf("195.4"), mp.mpf("6060."), mp.mpf("20.08"), mp.mpf("5.62"),
                      mp.mpf("20.64"), mp.mpf("1.43"))
    return (mp.mpf("14.82") - 4705 / T) * LN10 + 2 * mp.log(mp.mpf(101325))


def qp(A, b, C, d):
    n2, n3 = len(b), len(d)
    A, b, C, d = mp.matrix(A), mp.matrix(b), mp.matrix(C), mp.matrix(d)
    H, g = A.T * A, A.T * b
    rows = [i for i in range(n3) if any(C[i, j] != 0 for j in range(n2))]
    scale = max([abs(x) for x in d] + [mp.mpf(1e-300)])
    cands = []
    for k in range(0, n2 + 1):
        for S in itertools.combinations(rows, k):
            K = mp.zeros(n2 + k, n2 + k)
            r = mp.zeros(n2 + k, 1)
            for i in range(n2):
                for j in range(n2):
                    K[i, j] = H[i, j]
                r[i] = g[i]
            for a, i in enumerate(S):
                for j in range(n2):
                    K[n2 + a, j] = C[i, j]
                    K[j, n2 + a] = C[i, j]
                r[n2 + a] = d[i]
            try:
                z = mp.lu_solve(K, r)
            except ZeroDivisionError:
                continue
            x = [z[i] for i in range(n2)]
            viol = max([sum(C[i, j] * x[j] for j in range(n2)) - d[i] for i in rows] + [mp.mpf(0)])
            res = A * mp.matrix(x) - b
            obj = sum(v * v for v in res)
            # KKT: A^T (A x - b) + C_S^T lam = 0 with lam >= 0 at the optimum; z holds lam
            lam = [z[n2 + a] for a in range(k)]
            cands.append(dict(S=S, x=x, obj=obj, viol=viol / scale, lam=lam))
    # same allowance as the solver's own origin check: 64 eps of max|d|
    feas = [c for c in cands if c["viol"] <= 64 * mp.mpf(2) ** -52]
    best = min(feas, key=lambda c: c["obj"])
    return best, feas


def objective(A, b, x):
    A, b = mp.matrix(A), mp.matrix(b)
    r = A * mp.matrix([mp.mpf(v) for v in x]) - b
    return sum(v * v for v in r)


def violation(C, d, x):
    n2 = len(x)
    return max(sum(mp.mpf(C[i][j]) * mp.mpf(x[j]) for j in range(n2)) - mp.mpf(d[i]) for i in range(len(d)))


# --- equilibrium -----------------------------------------------------------------------
# species: 0 dry, 1 H2O, 2 NH3, 3 H2S, 4 H2O(l), 5 H2O(l,p), 6 NH3(s), 7 NH3(s,p), 8 NH4SH(s)

def _split(c, logs, mask):
    W, N, S = c[1] + c[4], c[2] + c[6] + c[8], c[3] + c[8]
    g = [mp.e ** l if l is not None else mp.mpf(0) for l in logs]
    c4 = W - g[0] if mask[0] else mp.mpf(0)
    c8 = S - g[2] if mask[2] else mp.mpf(0)
    c6 = N - g[1] - c8 if mask[1] else mp.mpf(0)
    return g, c4, c6, c8, (W, N, S)


def _resid(T, g, c4, c6, c8, tot, mask, lp, live):
    W, N, S = tot
    f = []
    if live[0]:
        f.append(lp[0] - lsvp(0, T) if mask[0] else g[0] / W - 1)
    if live[1]:
        f.append(lp[1] - lsvp(1, T) if mask[1] else (g[1] + c8) / N - 1)
    if live[2]:
        f.append(lp[1] + lp[2] - lsvp(2, T) if mask[2] else g[2] / S - 1)
    return f


def _admissible(T, g, c4, c6, c8, mask, lp, tol=1e-9):
    for m, a in zip(mask, [c4, c6, c8]):
        if m and a < 0:
            return False
    ninf = mp.mpf("-1e400")
    l = [v if v is not None else ninf for v in lp]
    sat = [l[0] - lsvp(0, T), l[1] - lsvp(1, T), l[1] + l[2] - lsvp(2, T)]
    return all(m or s_ <= tol for m, s_ in zip(mask, sat))


def _solve(c, T0, mask, ptot, uv=None):
    """Gases with a zero element total are fixed at zero, and the clouds that need them are
    absent; the remaining unknowns are the log amounts of the live gases (and T for UV)."""
    c = [mp.mpf(v) for v in c]
    W, N, S = c[1] + c[4], c[2] + c[6] + c[8], c[3] + c[8]
    live = [W > 0, N > 0, S > 0]
    if (mask[0] and not live[0]) or ((mask[1] or mask[2]) and not live[1]) or (mask[2] and not live[2]):
        return None
    tot = (W, N, S)
    for fr in ("0.5", "0.9", "0.01", "1e-6", "1e-12"):
        f = mp.mpf(fr)
        s0 = [mp.log(W * (f if mask[0] else 1)) if live[0] else None,
              mp.log(N * (f if (mask[1] or mask[2]) else 1)) if live[1] else None,
              mp.log(S * (f if mask[2] else 1)) if live[2] else None]
        idx = [k for k in range(3) if live[k]]

        def unpack(v):
            logs = [None, None, None]
            for n, k in enumerate(idx):
                logs[k] = v[n]
            T = v[len(idx)] if uv else mp.mpf(T0)
            return logs, T

        def F(*v):
            logs, T = unpack(v)
            g, c4, c6, c8, _ = _split(c, logs, mask)
            lp = ptot(T, g, c4, c6, c8)
            f = _resid(T, g, c4, c6, c8, tot, mask, lp, live)
            if uv:
                a, b, U = uv
                cc = [c[0], g[0], g[1], g[2], c4, c[5], c6, c[7], c8]
                f.append(sum(ci * (ai + bi * T) for ci, ai, bi in zip(cc, a, b)) / U - 1)
            return f
        x0 = [s0[k] for k in idx] + ([mp.mpf(T0)] if uv else [])
        try:
            sol = mp.findroot(F, x0, tol=mp.mpf(10) ** -(mp.mp.dps - 12), maxsteps=200)
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
        sol = [sol[n] for n in range(len(x0))] if len(x0) > 1 else [sol]
        logs, T = unpack(sol)
        g, c4, c6, c8, _ = _split(c, logs, mask)
        lp = ptot(T, g, c4, c6, c8)
        return dict(T=T, c=[c[0], g[0], g[1], g[2], c4, c[5], c6, c[7], c8], lp=lp,
                    ok=_admissible(T, g, c4, c6, c8, mask, lp))
    return "nosol"


def eq_uv(c, T0, a, b, U):
    def ptot(T, g, c4, c6, c8):
        return [mp.log(gi * R * T) if gi > 0 else None for gi in g]
    out = {}
    for mask in itertools.product((0, 1), repeat=3):
        out[mask] = _solve(c, T0, mask, ptot, uv=([mp.mpf(v) for v in a], [mp.mpf(v) for v in b], mp.mpf(U)))
    return out


def eq_tp(x, T, P):
    P = mp.mpf(P)
    x0 = [mp.mpf(v) for v in x]

    def ptot(T, g, c4, c6, c8):
        ng = x0[0] + g[0] + g[1] + g[2]
        return [mp.log(P * gi / ng) if gi > 0 else None for gi in g]
    out = {}
    for mask in itertools.product((0, 1), repeat=3):
        r = _solve(x, T, mask, ptot)
        if isinstance(r, dict):
            tot = sum(r["c"])
            r["x"] = [ci / tot for ci in r["c"]]
        out[mask] = r
    return out
