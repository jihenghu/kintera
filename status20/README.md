# status20: kintera status-20 package

Status-20 (max_iter stop) cells of the tests/jupiter.yaml sweeps at two kintera commits:

- base: a40a73d331f487cd782fbff49c887505733dbf8b
- head: 6ed29f1d7c6af45eaba92c7027689f5529fb2044 (UCzhangxi/kintera PR #93)

## Files

- `kintera-status20-package.md`: the full package, identical to the copy posted in the thread. It covers state generation, scripts, build configuration, status-20 counts and the per-cell tables.
- `scripts/sweep.py`: the sweep (wide and cold families, 5000 random states each plus the PR cell #5000; TP and UV(kkt) at max_iter 100 and 1000, UV(auto) at 100).
- `scripts/brute.py`, `scripts/ucoef.py`, `scripts/outer.py`: the independent brute-force reference, which tries all 8 condensate assemblages (H2O(l), NH3(s), NH4SH(s)) ∈ {0,1}³.
- `csv/status20_maxiter100.csv`: every cell that is status 20 at max_iter 100 at base or head (781 rows). Section 6a of the .md.
- `csv/status20_maxiter1000.csv`: the same for max_iter 1000 (719 rows), given as differences from the max_iter 100 row. Section 6b of the .md.

The scripts and CSVs are copied verbatim from the code blocks in the .md. The column definitions are in section 6 of the .md.

## Running

Placeholders: `<KINTERA>` is a kintera checkout at one of the two commits; `<DECK>` is `<KINTERA>/tests/jupiter.yaml`; `<PYINSTALL_CPU>` / `<PYINSTALL_CUDA>` are installs made with `pip install --no-build-isolation --no-deps -t <PYINSTALL> .` from a CPU or CUDA build; `<OUT>` is an output folder.

```bash
cd status20/scripts
CUDA_VISIBLE_DEVICES= python -I sweep.py <PYINSTALL_CPU>  <DECK> cpu  <OUT>/sw-base-cpu.pt
CUDA_VISIBLE_DEVICES=0 python -I sweep.py <PYINSTALL_CUDA> <DECK> cuda <OUT>/sw-base-gpu0.pt
# repeat with the 6ed29f1d installs -> <OUT>/sw-head-cpu.pt, <OUT>/sw-head-gpu0.pt

# brute force (needs mpmath; uses a Pool of 28 processes). Pass the head file as the 4th argument.
python outer.py <DECK> <PYINSTALL_CPU> <OUT>/sw-head-cpu.pt <OUT>/sw-head-cpu.pt <OUT>/sw-base-cpu.pt <OUT>/outer-cpu.pt
```

`outer.py` contains the literal `"<SCRIPTDIR>"` on its sys.path line. Running it from `scripts/` works because Python already puts the script's own folder on the path. To call the references directly, use `brute.eq_tp(x, T, P)` or `brute.eq_uv(c, T0, a, b, U)` with `a, b` from `ucoef.ucoef(ThermoY(...))`. Section 3 of the .md has the details.
