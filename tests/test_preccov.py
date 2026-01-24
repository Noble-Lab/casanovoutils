# test_graph_prec_cov.py
#
# Pytest tests for GraphPrecCov.
#
# - Uses matplotlib "Agg" backend (no GUI)
# - Monkeypatches get_ground_truth / prec_cov so we don't depend on real files
# - Verifies:
#   * __post_init__/clear sets up axes limits/labels/title
#   * add_peptides calls helpers with correct args and plots correct data/label
#   * save writes a file
#   * add_amino_acids raises NotImplementedError
#
# IMPORTANT: change the import to match your package/module path.

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib

matplotlib.use("Agg")  # headless backend for CI


# CHANGE THIS to your real module, e.g.
# from casanovoutils.graph_prec_cov import GraphPrecCov
from casanovoutils.preccov import GraphPrecCov


@pytest.fixture
def dummy_predictions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pep_score": np.array([0.2, 0.9, 0.5], dtype=float),
            "pep_correct": np.array([False, True, True], dtype=bool),
        }
    )


def test_init_calls_clear_and_sets_axes_defaults():
    g = GraphPrecCov(
        fig_width=4.0,
        fig_height=2.5,
        fig_dpi=123,
        ax_x_label="Cov",
        ax_y_label="Prec",
        ax_title="MyTitle",
    )

    # Ax exists and is configured
    assert g.fig is not None
    assert g.ax is not None

    # Limits
    assert g.ax.get_xlim() == (0.0, 1.0)
    assert g.ax.get_ylim() == (0.0, 1.0)

    # Labels/title (note: code appends "(Amino Acid)")
    assert g.ax.get_xlabel() == "Cov"
    assert g.ax.get_ylabel() == "Prec"
    assert g.ax.get_title() == "MyTitle (Amino Acid)"


def test_clear_resets_figure_and_axes_objects():
    g = GraphPrecCov()
    old_fig, old_ax = g.fig, g.ax

    g.clear()
    assert g.fig is not old_fig
    assert g.ax is not old_ax

    # still configured
    assert g.ax.get_xlim() == (0.0, 1.0)
    assert g.ax.get_ylim() == (0.0, 1.0)


def test_add_peptides_calls_helpers_and_plots(monkeypatch, dummy_predictions_df):
    g = GraphPrecCov()

    calls = {"get_ground_truth": [], "prec_cov": []}

    def fake_get_ground_truth(mztab_path, mgf_path, replace_i_l=False):
        calls["get_ground_truth"].append((mztab_path, mgf_path, replace_i_l))
        return dummy_predictions_df

    # Return a curve where we can assert exact values were plotted
    fake_prec = np.array([1.0, 0.5, 2 / 3], dtype=float)
    fake_cov = np.array([1 / 3, 2 / 3, 1.0], dtype=float)
    fake_aupc = 0.42

    def fake_prec_cov(scores, is_correct):
        # record that inputs were to_numpy() arrays
        calls["prec_cov"].append((scores.copy(), is_correct.copy()))
        return fake_prec, fake_cov, fake_aupc

    import casanovoutils.preccov as mod

    monkeypatch.setattr(mod, "get_ground_truth", fake_get_ground_truth)
    monkeypatch.setattr(mod, "prec_cov", fake_prec_cov)

    g.add_peptides("x.mztab", "y.mgf", "ModelX", replace_i_l=True)

    # Helper calls
    assert calls["get_ground_truth"] == [("x.mztab", "y.mgf", True)]
    assert len(calls["prec_cov"]) == 1

    got_scores, got_is_correct = calls["prec_cov"][0]
    assert np.allclose(got_scores, dummy_predictions_df["pep_score"].to_numpy())
    assert np.array_equal(
        got_is_correct, dummy_predictions_df["pep_correct"].to_numpy()
    )

    # One line added with x=cov, y=prec
    assert len(g.ax.lines) == 1
    line = g.ax.lines[0]
    assert np.allclose(line.get_xdata(), fake_cov)
    assert np.allclose(line.get_ydata(), fake_prec)

    # Legend label includes AUPC formatted to 3 decimals
    assert line.get_label() == "ModelX 0.420"

    # Legend exists and uses lower left (hard-coded in code)
    leg = g.ax.get_legend()
    assert leg is not None
    assert [t.get_text() for t in leg.get_texts()] == ["ModelX 0.420"]


def test_save_writes_file(tmp_path):
    g = GraphPrecCov()
    out = tmp_path / "plot.png"

    g.save(out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_show_calls_fig_show(monkeypatch):
    g = GraphPrecCov()
    called = {"n": 0}

    def fake_show():
        called["n"] += 1

    monkeypatch.setattr(g.fig, "show", fake_show)
    g.show()
    assert called["n"] == 1
