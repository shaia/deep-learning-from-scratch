"""
Agreement + gradient tests for Module 07 (LSTM).

Three gates, all required to pass:

1. C <-> Python agreement. Compile the C toy (linking the shared nanograd
   library, now with the LSTM), run it, parse its FINAL lines, run the Python
   mirror (ToyLatch) on the same seeds, and assert every tracked value agrees.
   Both train four networks on the latch -- Module 06's RNN, an LSTM with open
   forget and shut input gates, the same LSTM with the 1997 cell-only
   gradient, and the same weights with the textbook gate biases -- on the same
   batches, in the same op order from the same LCG streams, with the C
   library's exp and tanh on both sides, so they match to the last digit in
   practice, comfortably inside the 1e-9 gate. One more FINAL line pins the C
   LSTM forward/backward (initial state and cell, input gradients, truncation,
   the cell-only gradient) to nanograd's.

2. Finite-difference gradient check of the NEW nanograd machinery -- the
   honesty gate. The LSTM's dX, dh0, dc0, dWx, dWh and db against finite
   differences; its truncated backward against what truncation MEANS (the sum
   of full-BPTT gradients of the chunks, each starting from the state AND cell
   the previous chunk left, held fixed); its cell-only backward against what
   THAT means (the exact gradient of a network whose gates read a frozen copy
   of h<t-1>, so that error can only travel back along the cell); and small
   latch and character networks end to end through softmax cross-entropy,
   every parameter.

3. nanograd (NumPy) vs the C-order ToyLatch. The vectorized library and the
   mirror must compute the same initial loss and gradients for every twin, and
   the same parameters after three Adam steps on the same batches.

Run:
    python topics/07-lstm/tests/test_agreement.py

Exits 0 and prints "OK" on success; non-zero with a diff on failure. Plain
asserts, no test framework -- runs anywhere Python + clang do.
"""

import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOPIC = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(TOPIC))
C_DIR = os.path.join(TOPIC, "c")
PY_DIR = os.path.join(TOPIC, "python")
LIB_C = os.path.join(REPO, "lib", "c", "nanograd")
LIB_PY = os.path.join(REPO, "lib", "python")

TOL = 1e-9         # C<->Python: same op order, same libm.
GRAD_TOL = 1e-6    # analytic vs numerical gradient: a correct derivation is ~1e-8.
EXACT_TOL = 1e-12  # fast path vs definition: same arithmetic, different order.

# Make the Python mirror, the shared library, and the check utilities importable.
sys.path.insert(0, PY_DIR)
sys.path.insert(0, LIB_PY)
sys.path.insert(0, HERE)
import lstm as py  # noqa: E402
from check_utils import (  # noqa: E402
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    rel_error,
)
from nanograd import LSTM, Adam, SoftmaxCrossEntropy, clip_grad_norm  # noqa: E402
from nanograd.lstm import _sigmoid  # noqa: E402
from nanograd.rng import Rng  # noqa: E402

C_SOURCES = ("rng.c", "nn.c", "init.c", "optim.c", "rnn.c", "lstm.c")


def compile_c() -> str:
    """Compile the C latch toy + nanograd sources; return the binary path."""
    cc = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        print("SKIP: no C compiler (clang/gcc/cc) found on PATH")
        sys.exit(0)
    out_dir = tempfile.mkdtemp(prefix="lstm_")
    # The exe is run after this function returns, so clean up at exit.
    atexit.register(shutil.rmtree, out_dir, ignore_errors=True)
    exe = os.path.join(out_dir, "lstm.exe" if os.name == "nt" else "lstm")
    sources = [os.path.join(C_DIR, "lstm.c")] + [
        os.path.join(LIB_C, f) for f in C_SOURCES]
    cmd = [cc, "-O2", "-std=c11", "-Wall", "-Wextra", "-I", LIB_C, *sources, "-o", exe]
    # On Windows math is in the CRT; only POSIX toolchains need -lm.
    if os.name != "nt":
        cmd.append("-lm")
    subprocess.run(cmd, check=True)
    return exe


def parse_c_output(text: str):
    """Extract {'rnn': {...}, 'lstm': {...}, ...} from the FINAL lines."""
    results = {}
    for line in text.splitlines():
        m = re.match(r"FINAL (\w+) (.*)", line)
        if not m:
            continue
        pairs = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
        results[m.group(1)] = {k: float(v) for k, v in pairs.items()}
    return results


def close(a, b, tol=TOL):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def compare(label, c_vals, py_pairs):
    """Print and check one FINAL line against its Python fingerprint."""
    py_fp = dict(py_pairs)
    missing = sorted(set(py_fp) - set(c_vals))
    assert not missing, f"C FINAL {label} line is missing keys {missing}"
    ok = True
    for name, pv in py_fp.items():
        cv = c_vals[name]
        good = close(cv, pv)
        print(f"  {name:10s}  C={cv:.17g}  Py={pv:.17g}  {'ok' if good else 'MISMATCH'}")
        if not good:
            print(f"    -> diff={abs(cv - pv):.3e}")
        ok = ok and good
    return ok


def check_agreement():
    """Gate 1: C and Python train all four latch twins (and run the LSTM
    check) identically."""
    exe = compile_c()
    proc = subprocess.run([exe], capture_output=True, text=True, check=True)
    c = parse_c_output(proc.stdout)
    for key in py.TWINS + ("cell",):
        assert key in c, f"missing FINAL {key} line:\n{proc.stdout}"

    ok = True
    tests = py.test_sets()
    titles = {"rnn": "Module 06's RNN, full BPTT",
              "lstm": "LSTM, forget bias +5 / input bias -5, full BPTT",
              "carousel": "the same LSTM, the 1997 cell-only gradient",
              "forget1": "the same weights, the textbook biases"}
    for twin in py.TWINS:
        print(f"== C <-> Python agreement (the latch, T={py.T_TOY}: {titles[twin]}) ==")
        ok &= compare(twin, c[twin], py.run_toy(twin, tests=tests))
    print("== C <-> Python agreement (LSTM forward/backward: h0, c0, dX, truncation, "
          "cell-only) ==")
    ok &= compare("cell", c["cell"], py.lstm_fingerprint())
    if ok:
        print(f"  -> all values agree within tol={TOL:g}")
    return ok


def _grad_line(name, analytic, numeric, tol=GRAD_TOL):
    err = rel_error(analytic, numeric)
    good = err < tol
    print(f"  {name:38s} rel_error={err:.3e}  {'ok' if good else 'MISMATCH'}")
    return good


def _vec_error(analytic, numeric):
    """||analytic - numeric|| / ||analytic||: one parameter's gradient as a vector."""
    return float(np.linalg.norm(analytic - numeric) / max(1e-12, np.linalg.norm(analytic)))


def _random_lstm(rng, n_in, n_h, scale=0.6):
    layer = LSTM(n_in, n_h)
    for p in layer.params():
        p[...] = rng.standard_normal(p.shape) * scale
    return layer


def _frozen_forward(layer, X, c0, H_read):
    """An LSTM whose gates read H_read[t] -- a constant -- in place of
    h<t-1>. Its exact gradient is what cell_only computes: error reaches the
    past only through c."""
    n = layer.n_h
    c, out = c0, []
    for t in range(X.shape[0]):
        z = X[t] @ layer.Wx + H_read[t] @ layer.Wh + layer.b
        i, f = _sigmoid(z[:, :n]), _sigmoid(z[:, n:2 * n])
        g, o = np.tanh(z[:, 2 * n:3 * n]), _sigmoid(z[:, 3 * n:])
        c = f * c + i * g
        out.append(o * np.tanh(c))
    return np.array(out)


def check_gradients():
    """Gate 2: the LSTM's backward matches finite differences, truncation
    matches chunked training, the cell-only gradient matches its definition,
    and small networks are correct end to end."""
    print("\n== finite-difference gradient check (LSTM machinery) ==")
    ok = True
    rng = np.random.default_rng(0)

    # (a) The layer with an initial state and cell: every input and parameter.
    T, m, n_in, n_h = 7, 3, 4, 5
    layer = _random_lstm(rng, n_in, n_h)
    X = rng.standard_normal((T, m, n_in))
    h0 = rng.standard_normal((m, n_h)) * 0.5
    c0 = rng.standard_normal((m, n_h)) * 0.5
    dH = rng.standard_normal((T, m, n_h))

    def f(_):
        return layer.forward(X, h0, c0)

    f(None)
    dX = layer.backward(dH)
    ok &= _grad_line("LSTM dX", dX, eval_numerical_gradient_array(f, X, dH))
    for name, arr, got in (("dh0", h0, lambda: layer.dh0), ("dc0", c0, lambda: layer.dc0),
                           ("dWx", layer.Wx, lambda: layer.dWx),
                           ("dWh", layer.Wh, lambda: layer.dWh),
                           ("db", layer.b, lambda: layer.db)):
        f(None)
        layer.backward(dH)
        ok &= _grad_line(f"LSTM {name}", got().copy(), eval_numerical_gradient_array(f, arr, dH))

    # (b) Truncation is chunked training: the truncated backward equals the sum
    # of each chunk's full-BPTT gradients, each chunk starting from the state
    # and cell the previous one left (constants -- no gradient flows into them).
    for k in (1, 2, 3):
        H = layer.forward(X, h0, c0)
        C = layer.C.copy()
        layer.backward(dH, truncate=k)
        got = [g.copy() for g in layer.grads()]
        want = [np.zeros_like(g) for g in got]
        piece = LSTM(n_in, n_h)
        piece.Wx, piece.Wh, piece.b = layer.Wx, layer.Wh, layer.b
        for s in range(0, T, k):
            piece.forward(X[s:s + k], h0 if s == 0 else H[s - 1], c0 if s == 0 else C[s - 1])
            piece.backward(dH[s:s + k])
            want = [w + g for w, g in zip(want, piece.grads())]
        err = max(float(np.max(np.abs(a - b))) for a, b in zip(got, want))
        good = err < EXACT_TOL and np.all(layer.dh0 == 0.0) and np.all(layer.dc0 == 0.0)
        print(f"  {'truncate=' + str(k) + ' == chunked BPTT':38s} max|diff|={err:.3e}  "
              f"{'ok' if good else 'MISMATCH'}")
        ok &= good

    # (c) The cell-only gradient (1997) is the exact gradient of a network whose
    # gates read a frozen h<t-1>: finite differences of _frozen_forward.
    H = layer.forward(X, h0, c0)
    H_read = np.concatenate([h0[None], H[:-1]])
    layer.backward(dH, cell_only=True)
    got = {"dWx": layer.dWx.copy(), "dWh": layer.dWh.copy(), "db": layer.db.copy(),
           "dc0": layer.dc0.copy()}
    ok &= _grad_line("cell-only dc0 (frozen-h network)", got["dc0"], eval_numerical_gradient_array(
        lambda _: _frozen_forward(layer, X, c0, H_read), c0, dH))
    for name, arr in (("dWx", layer.Wx), ("dWh", layer.Wh), ("db", layer.b)):
        ok &= _grad_line(f"cell-only {name} (frozen-h network)", got[name],
                         eval_numerical_gradient_array(
                             lambda _: _frozen_forward(layer, X, c0, H_read), arr, dH))
    good = np.all(layer.dh0 == 0.0)
    print(f"  {'cell-only: nothing reaches h0':38s} {'ok' if good else 'MISMATCH'}")
    ok &= good

    # (d) Small networks end to end: every parameter's hand-written gradient
    # vs the numerical one, through the read-out and softmax cross-entropy.
    # Compared as whole vectors: a network's gradient entries span six orders
    # of magnitude (an early step's share is scaled by every forget gate after
    # it), and central differences resolve an entry only to ~1e-11 absolute,
    # so the entrywise rel_error of a 1e-7 entry measures the noise floor, not
    # the derivation. The layer itself was checked entrywise in (a).
    loss_layer = SoftmaxCrossEntropy()
    lay, head = py.build_latch_net("lstm", Rng(5), 5, {"i": 0.3, "f": 0.8})
    S = py.latch_batch(np.random.default_rng(1), 9, 3)
    Xl = py.one_hot(S)

    def latch_loss():
        return py.latch_loss_and_grads(lay, head, Xl, S, loss_layer=loss_layer)

    latch_loss()
    worst = max(_vec_error(g, eval_numerical_gradient(lambda _: latch_loss(), p))
                for p, g in zip(lay.params() + head.params(),
                                [g.copy() for g in lay.grads() + head.grads()]))
    good = worst < GRAD_TOL
    print(f"  {'latch net (T = 9), all 5 params':38s} worst |a-n|/|a|={worst:.3e}  "
          f"{'ok' if good else 'MISMATCH'}")
    ok &= good

    data = py.load_text_data()
    cm = py.CharLSTM(len(data[2]), 6, seed=3, dtype=np.float64)
    ids = data[1][:6 * 4 + 1]
    xs, ys = ids[:-1].reshape(4, 6).T, ids[1:].reshape(4, 6).T
    st = (np.random.default_rng(4).standard_normal((4, 6)) * 0.3,
          np.random.default_rng(5).standard_normal((4, 6)) * 0.3)

    def char_loss():
        return cm.loss_and_grads(xs, ys, st)[0]

    char_loss()
    worst = max(_vec_error(g, eval_numerical_gradient(lambda _: char_loss(), p))
                for p, g in zip(cm.params(), [g.copy() for g in cm.grads()]))
    good = worst < GRAD_TOL
    print(f"  {'char LSTM (h0, c0 carried), all 5 params':38s} worst |a-n|/|a|={worst:.3e}  "
          f"{'ok' if good else 'MISMATCH'}")
    ok &= good

    if ok:
        print(f"  -> all gradients match numerically within {GRAD_TOL:g}; truncation equals "
              f"chunked training within {EXACT_TOL:g}")
    return ok


def check_paths_agree():
    """Gate 3: vectorized nanograd and the C-order ToyLatch agree on the
    initial loss and gradients and on every parameter after three Adam steps,
    for every twin."""
    print("\n== nanograd (NumPy) vs C-order ToyLatch ==")
    ok = True
    biases = {"rnn": None, "lstm": py.OPEN_BIAS, "carousel": py.OPEN_BIAS,
              "forget1": py.TEXTBOOK_BIAS}
    for twin in py.TWINS:
        kind = "rnn" if twin == "rnn" else "lstm"
        cell_only = twin == "carousel"
        mirror = py.ToyLatch(Rng(py.TOY_SEED), twin)
        layer, head = py.build_latch_net(kind, bias=biases[twin])
        S = py.draw_latch(Rng(py.DATA_SEED), py.M_TOY, py.T_TOY)
        X = py.one_hot(S)
        loss_m, grads_m = mirror.loss_and_grads(X, S)
        loss_v = py.latch_loss_and_grads(layer, head, X, S, cell_only)
        err_l = rel_error(loss_m, loss_v)
        err_g = max(rel_error(a, b) for a, b in zip(grads_m, layer.grads() + head.grads()))
        good = err_l < 1e-9 and err_g < 1e-9
        print(f"  {twin:8s} step 0: loss rel_error={err_l:.3e}  worst grad rel_error={err_g:.3e}  "
              f"{'ok' if good else 'MISMATCH'}")
        ok &= good

        # Three Adam steps on each path, on the same batches.
        mirror = py.ToyLatch(Rng(py.TOY_SEED), twin)
        layer, head = py.build_latch_net(kind, bias=biases[twin])
        opt = Adam(lr=py.TOY_LR, beta1=py.ADAM_B1, beta2=py.ADAM_B2, eps=py.ADAM_EPS)
        r_m, r_v = Rng(py.DATA_SEED), Rng(py.DATA_SEED)
        for _ in range(3):
            mirror.step(r_m)
            S = py.draw_latch(r_v, py.M_TOY, py.T_TOY)
            py.latch_loss_and_grads(layer, head, py.one_hot(S), S, cell_only)
            grads = layer.grads() + head.grads()
            clip_grad_norm(grads, py.CLIP)
            opt.step(layer.params() + head.params(), grads)
        err = max(rel_error(a, b) for a, b in zip(mirror.params(), layer.params() + head.params()))
        good = err < 1e-9
        print(f"  {twin:8s} after 3 steps: worst param rel_error={err:.3e}  "
              f"{'ok' if good else 'MISMATCH'}")
        ok &= good
    return ok


def main():
    a = check_agreement()
    g = check_gradients()
    p = check_paths_agree()
    if a and g and p:
        print("\nOK: C<->Python agree, the LSTM gradients are correct, "
              "and the two Python paths match.")
        sys.exit(0)
    print("\nFAIL: see mismatches above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
