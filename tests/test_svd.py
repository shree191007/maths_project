import numpy as np

from climatematrix import svd_analysis as sv


def _A(seed=0, shape=(8, 60)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(shape[0], 3)) @ rng.normal(size=(3, shape[1])) + rng.normal(size=(shape[0], 1))


def test_full_rank_reconstruction_is_exact():
    A = _A(); r = sv.compute_svd(A)
    assert np.allclose(sv.reconstruct(r, r.rank), A)


def test_eckart_young_error_equals_tail_singular_values():
    A = np.random.default_rng(1).normal(size=(8, 50)); r = sv.compute_svd(A)
    t = sv.rank_table(A, r)
    assert np.allclose(t["frob_error"], t["eckart_young"])


def test_error_decreases_monotonically_with_rank():
    A = np.random.default_rng(2).normal(size=(8, 50)); t = sv.rank_table(A, sv.compute_svd(A))
    assert (np.diff(t["frob_error"]) <= 1e-12).all()


def test_explained_variance_sums_to_one_and_sorted():
    r = sv.compute_svd(_A()); ev = sv.explained_variance(r.s)
    assert np.isclose(ev.sum(), 1) and (np.diff(ev) <= 0).all()


def test_exact_low_rank_has_numerical_rank():
    rng = np.random.default_rng(3)
    A = rng.normal(size=(8, 3)) @ rng.normal(size=(3, 40))
    assert sv.numerical_rank(np.linalg.svd(A, compute_uv=False)) == 3


def test_centering_modes():
    A = _A()
    assert np.allclose(sv.center(A, "row")[0].mean(axis=1), 0)
    assert np.allclose(sv.center(A, "col")[0].mean(axis=0), 0)


def test_weighted_svd_round_trips():
    A = _A(); w = np.sqrt(np.linspace(0.5, 2, 8)); r = sv.compute_svd(A, "row", w)
    assert np.allclose(sv.reconstruct(r, r.rank), A)


def test_hard_impute_recovers_exact_low_rank_matrix():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 146)) + rng.normal(size=(8, 1))
    m = sv.random_mask(A.shape, 0.2, rng)
    X = sv.impute_lowrank(np.where(m, np.nan, A), 2)
    assert np.sqrt(np.mean((X[m] - A[m]) ** 2)) < 1e-4


def test_impute_keeps_observed_entries_and_k0_is_row_mean():
    rng = np.random.default_rng(1); A = _A(shape=(8, 40)); m = sv.random_mask(A.shape, 0.2, rng)
    Ao = np.where(m, np.nan, A)
    X = sv.impute_lowrank(Ao, 0)
    assert np.allclose(X[~m], A[~m])
    assert np.allclose(X[m], np.broadcast_to(np.nanmean(Ao, axis=1, keepdims=True), A.shape)[m])


def test_soft_impute_with_large_lambda_collapses_to_row_mean():
    rng = np.random.default_rng(2); A = _A(shape=(8, 40)); m = sv.random_mask(A.shape, 0.2, rng)
    Ao = np.where(m, np.nan, A)
    assert np.allclose(sv.impute_soft(Ao, 1e6), sv.impute_lowrank(Ao, 0))


def test_masks_spare_edges():
    rng = np.random.default_rng(0)
    assert not sv.random_mask((8, 50), 0.5, rng)[:, [0, -1]].any()
    b = sv.block_mask((8, 50), 5, 2, rng)
    assert not b[:, [0, -1]].any() and (b.sum(axis=1) == 10).all()
