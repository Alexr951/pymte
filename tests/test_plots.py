import matplotlib
import pytest

import pymte

matplotlib.use("Agg")

AE_ARGS = dict(
    target="att",
    ivlike="worked ~ morekids + samesex + morekids*samesex",
    propensity="morekids ~ samesex + yob",
)


@pytest.fixture(scope="module")
def bounds_result():
    return pymte.ivmte(pymte.load_ae(), m0="~ u + yob", m1="~ u + yob", seed=0, **AE_ARGS)


@pytest.fixture(scope="module")
def point_result():
    sim = pymte.load_sim_data()
    with pytest.warns(UserWarning):
        return pymte.ivmte(
            sim, target="ate", m0="~ u", m1="~ u", ivlike="y ~ d + C(z)", propensity="d ~ C(z)"
        )


def test_plot_mtr_and_mte_bounds(bounds_result):
    ax = pymte.plot_mtr(bounds_result, at={"yob": 50})
    assert len(ax.get_lines()) == 4  # m0 and m1 at each bound
    ax2 = pymte.plot_mte(bounds_result, at={"yob": 50})
    assert len(ax2.get_lines()) == 3  # two bounds plus the zero line
    with pytest.raises(ValueError, match="pass their values with 'at'"):
        pymte.plot_mtr(bounds_result)


def test_plot_mtr_point(point_result):
    ax = pymte.plot_mtr(point_result)
    assert [line.get_label() for line in ax.get_lines()] == ["m1, estimate", "m0, estimate"]


def test_plot_weights(bounds_result, point_result):
    ax = pymte.plot_weights(bounds_result)
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels[0] == "target (att)" and len(labels) == 1 + 4
    ax2 = pymte.plot_weights(point_result)
    assert len(ax2.get_lines()) == 1 + 5


def test_ivmte_accepts_term_lists():
    ae = pymte.load_ae()
    terms = [(0, None), (1, None), (0, "yob")]
    r = pymte.ivmte(ae, m0=terms, m1=terms, seed=0, **AE_ARGS)
    assert r.bounds == pytest.approx((-0.1028836, -0.07818869), abs=1e-6)
