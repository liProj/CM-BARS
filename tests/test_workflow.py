from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from api import read_data, factors, model, save_posterior, load_posterior, predictions
from methods import RESPONSES, from_coded
from experiments import make_folds, condition_groups, _key
from metrics import crps_censored_samples, zero_balanced_accuracy
from stats_tests import holm

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def data():
    return read_data(ROOT / "data/bbd_design_responses.csv")

def test_factor_units_round_trip(data):
    recovered = from_coded(factors(data))
    np.testing.assert_allclose(recovered, data[["pH", "additive_mM", "cnf_pct"]])

@pytest.mark.parametrize("protocol", ["LOCO", "RKF"])
def test_no_condition_overlap(data, protocol):
    groups = condition_groups(data)
    assert len(np.unique(groups)) == 13
    selected = [f for f in make_folds(data, n_rep=2) if f[0] == protocol]
    for _, _, _, train, test in selected:
        assert set(groups[train]).isdisjoint(groups[test])
    if protocol == "LOCO":
        assert len(selected) == 13
        assert sorted(len(f[-1]) for f in selected) == [1] * 12 + [3]
        assert sorted(np.concatenate([f[-1] for f in selected])) == list(range(15))

@pytest.mark.parametrize("column,value", [("pH", 9), ("tannic_acid", 100), ("acetosyringone", -1), ("cnf_pct", np.nan)])
def test_invalid_data_rejected(data, tmp_path, column, value):
    data.loc[0, column] = value
    path = tmp_path / "bad.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError): read_data(path)

def test_empirical_crps_with_zero_mass():
    x = np.array([0., 0., 2., 4.])
    expected = np.abs(x - 1).mean() - .5 * np.abs(x[:, None] - x[None, :]).mean()
    assert crps_censored_samples([1.], x[:, None]) == pytest.approx(expected)

def test_zero_event_scoring_uses_probability():
    assert zero_balanced_accuracy([0, 5], [2, 2], p_zero=[.8, .2]) == 1
    assert np.isnan(zero_balanced_accuracy([1, 5], [2, 2], p_zero=[.8, .2]))

def test_holm_reference_values():
    np.testing.assert_allclose(holm([.01, .02, .03, .04, .05]), [.05, .08, .09, .09, .09])

def test_cache_namespaces_are_isolated(monkeypatch, tmp_path):
    import experiments
    monkeypatch.setattr(experiments, "CACHE", str(tmp_path))
    monkeypatch.setenv("CMBARS_CACHE_NAMESPACE", "short")
    one = _key("LOCO", 0, 0, "CM-BARS")
    monkeypatch.setenv("CMBARS_CACHE_NAMESPACE", "paper")
    two = _key("LOCO", 0, 0, "CM-BARS")
    assert one != two

@pytest.mark.mcmc
def test_posterior_prediction_and_portable_round_trip(data, tmp_path):
    fitted = model(warmup=12, samples=16, chains=1, max_tree_depth=4)
    fitted.fit(factors(data), data[RESPONSES].to_numpy())
    q = data.iloc[:2]
    before = predictions(fitted, q, n_draw=40)
    assert before.shape[0] == 6
    assert np.isfinite(before.select_dtypes('number')).all().all()
    assert before['mean'].between(0, 100).all()
    assert before.q05.between(0, 100).all() and before.q95.between(0, 100).all()
    assert (before.q05 <= before.q95).all()
    assert before.p_zero.between(0, 1).all()
    destination = tmp_path / "posterior.npz"
    save_posterior(fitted, destination)
    restored = load_posterior(destination)
    pd.testing.assert_frame_equal(before, predictions(restored, q, n_draw=40))

def test_archived_paper_results():
    table = pd.read_csv(ROOT / "reference_results/cv_pooled.csv")
    loco = table[table.protocol == "LOCO"]
    means = loco.groupby('method').RMSE.mean()
    assert means['CM-BARS'] == pytest.approx(6.417, abs=.001)
    assert means['ExtraTrees'] == pytest.approx(5.646, abs=.001)
