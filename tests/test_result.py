import pytest

from reflexguard.result import CategoryResult, GuardResult, resolve_thresholds


def make(threshold=None):
    ranked = [
        CategoryResult("violence", 0.9, {"noul": 0.9}),
        CategoryResult("jailbreak", 0.6, {"noul": 0.6}),
        CategoryResult("pii", 0.1, {"noul": 0.1}),
    ]
    return GuardResult(ranked=ranked).at(threshold)


def test_no_threshold_ranks_only():
    res = make()
    assert res.violations is None
    assert res.flagged is None
    assert res.top.name == "violence"
    assert list(res.scores) == ["violence", "jailbreak", "pii"]
    assert res["pii"].flagged is None


def test_float_threshold():
    res = make(0.5)
    assert [c.name for c in res.violations] == ["violence", "jailbreak"]
    assert res.flagged is True
    assert res["pii"].flagged is False


def test_threshold_is_inclusive():
    assert make(0.6)["jailbreak"].flagged is True


def test_nothing_over_threshold():
    res = make(0.95)
    assert res.violations == []
    assert res.flagged is False


def test_dict_threshold():
    res = make({"violence": 0.95, "jailbreak": 0.5, "pii": 0.05})
    assert [c.name for c in res.violations] == ["jailbreak", "pii"]


def test_at_rethresholds_and_clears():
    res = make(0.5)
    assert [c.name for c in res.at(0.8).violations] == ["violence"]
    assert res.at(None).flagged is None
    assert res.flagged is True  # original unchanged


def test_getitem_unknown():
    with pytest.raises(KeyError):
        make()["nope"]


@pytest.mark.parametrize("bad", [-0.1, 1.1, "0.5", True])
def test_bad_float_threshold(bad):
    with pytest.raises(ValueError, match="threshold must be"):
        resolve_thresholds(bad, ["a"])


def test_dict_threshold_must_match_categories():
    with pytest.raises(ValueError, match="missing=\\['b'\\]"):
        resolve_thresholds({"a": 0.5}, ["a", "b"])
    with pytest.raises(ValueError, match="unknown=\\['z'\\]"):
        resolve_thresholds({"a": 0.5, "z": 0.5}, ["a"])
