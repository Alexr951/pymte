from ivmte import load_ae, load_sim_data


def test_ae_shape_and_columns():
    ae = load_ae()
    assert ae.shape == (209_133, 8)
    assert list(ae.columns) == [
        "worked",
        "hours",
        "morekids",
        "samesex",
        "yob",
        "black",
        "hisp",
        "other",
    ]
    assert set(ae["morekids"].unique()) == {0, 1}


def test_sim_data_shape_and_support():
    sim = load_sim_data()
    assert sim.shape == (5_000, 4)
    assert list(sim.columns) == ["y", "d", "z", "x"]
    assert set(sim["z"].unique()) == {0, 1, 2, 3}
    assert sim["x"].between(1, 10).all()
