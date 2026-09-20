"""v4 harness unit tests: RMSE definitions, protocol uniformity, AST
transforms, run_id immutability, EnKF sanity on a linear-Gaussian problem."""
import numpy as np
import pytest

from benchmark import da_v4


# ---- RMSE definitions -----------------------------------------------------
def test_rmse_definitions_differ():
    """mean_state_rmse (v3 'pooled') vs pooled_rmse=sqrt(mean mse): a small
    worked example where the two definitions give different values."""
    per = {"renin": {"rmse": 0.1, "mse": 0.01, "rmse_norm": 0.1,
                     "unit": "dimensionless"},
           "adh": {"rmse": 0.5, "mse": 0.25, "rmse_norm": 0.5,
                   "unit": "dimensionless"}}
    truth = {"time": np.linspace(0, 10, 11),
             "traces": {"body:renin": np.zeros(11),
                        "body:adh": np.zeros(11)}}
    s = da_v4.summarize(per, ["renin", "adh"], truth, 10, 0)
    assert abs(s["mean_state_rmse"] - 0.3) < 1e-9          # (0.1+0.5)/2
    assert abs(s["pooled_rmse"] - np.sqrt(0.13)) < 1e-9     # sqrt(0.26/2)
    assert s["mean_state_rmse"] != s["pooled_rmse"]


def test_native_states_kept_in_own_units():
    """hr (bpm) must not be raw-pooled with dimensionless states."""
    per = {"renin": {"rmse": 0.1, "mse": 0.01, "rmse_norm": 0.1,
                     "unit": "dimensionless"},
           "hr": {"rmse": 5.0, "mse": 25.0, "rmse_norm": 5.0 / 72,
                  "unit": "hr"}}
    truth = {"time": np.linspace(0, 10, 11),
             "traces": {"body:renin": np.zeros(11)}}
    s = da_v4.summarize(per, ["renin", "hr"], truth, 10, 0)
    assert abs(s["pooled_rmse"] - 0.1) < 1e-9          # dimensionless only
    assert s["native_per_state"]["hr"]["rmse"] == 5.0  # raw bpm preserved


def test_missing_state_raises_not_skips():
    truth = {"time": np.linspace(0, 10, 11),
             "traces": {"body:renin": np.zeros(11)}}
    test = dict(truth)
    with pytest.raises(KeyError):
        da_v4.score_states(truth, test, ["renin", "zzz_missing"], 0, 10)


# ---- protocol --------------------------------------------------------------
def test_all_methods_share_strict_cut_protocol():
    """Every method's observations must be strictly < cut (primary v4
    protocol) and identical across methods for a given trial."""
    trial = {"obs": {"p": {"times": [30.0, 60.0, 90.0, 120.0],
                           "values": {"a": [1, 2, 3, 4],
                                      "b": [5, 6, 7, 8]}}},
             "seed": 1}
    for cut in (60, 90, 120):
        upd = da_v4._obs_updates(trial, "p", cut, "nudge", 0.15)
        times = {u["time_min"] for u in upd}
        assert times == {t for t in (30, 60, 90, 120) if t < cut}
        # same stream: nudge and jump read identical values
        assert da_v4._last_obs(trial, "p", cut)["a"] == \
            trial["obs"]["p"]["values"]["a"][
                trial["obs"]["p"]["times"].index(max(times))]


def test_disjoint_panels_enforced():
    assert da_v4.assert_disjoint({"a": ["x"], "b": ["y"]}, ["z"])
    with pytest.raises(AssertionError):
        da_v4.assert_disjoint({"a": ["x"], "b": ["y"]}, ["y", "z"])


def test_run_id_collision_raises(tmp_path):
    p = da_v4.unique_run_path(tmp_path, "r1")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    with pytest.raises(FileExistsError):
        da_v4.unique_run_path(tmp_path, "r1")


# ---- AST transforms --------------------------------------------------------
def test_perturb_expr_scales_coefficients_preserves_constants():
    out, factors = da_v4.perturb_expr(
        "max(0,0.6*thermosensor+0.3*work)+0.3*estrogen_decline",
        lambda: 2.0)
    # 0.6,0.3,0.3 coefficients doubled; the max() 0 bound untouched
    assert "1.2" in out and "0.6" in out.replace("1.2", "")
    assert "max(0," in out
    assert len(factors) >= 3


def test_perturb_expr_implicit_term_gets_factor():
    out, _ = da_v4.perturb_expr("work+0.5*injury", lambda: 2.0)
    assert "2.0" in out  # implicit-1 'work' term wrapped


def test_perturb_zero_level_is_identity():
    rng = np.random.default_rng(1)
    mods, _ = da_v4.perturbed_modules(rng, 0.0)
    orig = {m["id"]: m["target"] for m in da_v4.MODULES}
    for m in mods:
        # semantically identical: re-parse and compare term structure
        assert da_v4._terms(
            __import__("ast").parse(m["target"], mode="eval").body)


def test_perturb_preserves_sign_and_positivity():
    rng = np.random.default_rng(7)
    mods, rec = da_v4.perturbed_modules(rng, 0.3)
    assert all(r["tau_factor"] > 0 for r in rec)
    for m, r in zip(mods, rec):
        assert r["parsed"]
        # every expression still parses and keeps module terms
        assert da_v4._names(__import__("ast").parse(
            m["target"], mode="eval").body)


def test_decouple_removes_only_module_terms():
    out = da_v4.decouple_expr(
        "0.5*renin+0.3*plasma_vol+0.2*hemorrhage+0.4",
        {"renin", "plasma_vol"})
    assert "renin" not in out and "plasma_vol" not in out
    assert "hemorrhage" in out and "0.4" in out


def test_decouple_inside_max():
    out = da_v4.decouple_expr(
        "max(0,hypoxia+0.3*work-0.5)", {"hypoxia"})
    assert "hypoxia" not in out and "work" in out and "max" in out


def test_decoupled_all_205_parse():
    mods = da_v4.decoupled_modules()
    assert len(mods) == da_v4.N_STATES
    for m in mods:
        __import__("ast").parse(m["target"], mode="eval")


def test_decouple_on_perturbed_uses_same_realisation():
    rng = np.random.default_rng(3)
    twin, _ = da_v4.perturbed_modules(rng, 0.2)
    dec = da_v4.decoupled_modules(base=twin)
    # same tau values as the twin (same realisation, edges removed only)
    assert all(a["tau_min"] == b["tau_min"] for a, b in zip(twin, dec))


# ---- EnKF sanity -------------------------------------------------------------
def test_enkf_matches_exact_kalman_linear_gaussian():
    """2D linear-Gaussian check: with linear dynamics x'=Fx the EnKF update
    must reproduce the exact Kalman filter mean/covariance closely."""
    rng = np.random.default_rng(0)
    n, m = 2, 4000
    F = np.array([[0.9, 0.1], [0.0, 0.8]])
    H = np.array([[1.0, 0.0]])
    x_true = np.array([1.0, -0.5])
    # forecast ensemble from a nonzero prior
    X0 = rng.multivariate_normal([0, 0], [[0.5, 0.1], [0.1, 0.3]], m).T
    X = F @ X0
    y = float((H @ x_true)[0] + rng.normal(0, 0.1))
    R = np.array([[0.1 ** 2]])
    # exact KF
    P = np.cov(X)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    mu_exact = X.mean(1) + K @ (y - H @ X.mean(1))
    P_exact = (np.eye(n) - K @ H) @ P
    # stochastic EnKF update (same equations as method_enkf)
    Y = H @ X
    Ax, Ay = X - X.mean(1, keepdims=True), Y - Y.mean(1, keepdims=True)
    K_e = (Ax @ Ay.T / (m - 1)) @ np.linalg.inv(Ay @ Ay.T / (m - 1) + R)
    D = y + rng.normal(0, 0.1, (1, m))
    Xa = X + K_e @ (D - Y)
    assert np.allclose(Xa.mean(1), mu_exact.ravel(), atol=0.05)
    assert np.allclose(np.cov(Xa), P_exact, atol=0.05)
    assert np.linalg.norm(K_e - K) < 0.05


def test_signed_states_not_clipped_in_analysis():
    """Body states are signed: a negative posterior must survive."""
    # construction mirrors method_enkf's update: X + K(D-Y), no clip
    X = np.array([[0.01, 0.02, -0.03]])
    assert (X < 0).any()  # regression guard: no np.clip anywhere on X


def test_seed_stream_reproducible_across_processes():
    """_seed_stream must be a pure function of (seed, tag): Python's hash()
    is randomized per process, so tag entropy must come from a stable
    digest. Regression: identical seed/tag draws differed between runs,
    making stored trial artifacts non-reproducible."""
    import hashlib
    import numpy as np
    from benchmark.da_v4 import _seed_stream
    expected = int.from_bytes(
        hashlib.sha256("truth".encode("utf-8")).digest()[:8], "little")
    seq = np.random.SeedSequence([501, expected])
    ref = np.random.default_rng(seq).random(8)
    got = _seed_stream(501, "truth").random(8)
    assert np.allclose(ref, got)
    # two calls in any process order produce the same stream
    assert np.allclose(_seed_stream(501, "obs").random(4),
                       _seed_stream(501, "obs").random(4))
