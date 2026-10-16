import asyncio

import pytest

from reflexguard import CascadeGuard, EnsembleGuard, Guard, InputTooLong


class Fixed(Guard):
    """A guard with fixed per-category NOUL scores; can sleep, raise, or count calls."""

    def __init__(self, scores, threshold=0.5, delay=0.0, error=None, categories=None):
        super().__init__(categories=categories or list(scores), threshold=threshold)
        self.scores, self.delay, self.error, self.calls = scores, delay, error, 0

    async def predict(self, context, questions):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return {qid: {"noul": self.scores[qid.split(":", 1)[1]]} for qid in questions}


# ── EnsembleGuard ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("rule, k, flagged", [("any", None, {"violence", "pii"}), ("all", None, set()),
                                              ("votes", 2, {"violence"})])
async def test_ensemble_rules_combine_member_decisions(rule, k, flagged):
    a = Fixed({"violence": 0.9, "pii": 0.1})
    b = Fixed({"violence": 0.8, "pii": 0.2})
    c = Fixed({"violence": 0.1, "pii": 0.7})
    res = await EnsembleGuard([a, b, c], rule=rule, k=k).guard("x")
    assert {cat.name for cat in res.violations} == flagged


async def test_ensemble_score_is_the_share_of_members_flagging():
    a = Fixed({"violence": 0.9, "pii": 0.1})
    b = Fixed({"violence": 0.8, "pii": 0.2})
    c = Fixed({"violence": 0.1, "pii": 0.7})
    res = await EnsembleGuard([a, b, c], rule="votes", k=2).guard("x")
    assert res["violence"].score == pytest.approx(2 / 3) and res["violence"].threshold == pytest.approx(2 / 3)
    assert res["pii"].score == pytest.approx(1 / 3)
    assert res["violence"].by_mode == {"Fixed": 0.9, "Fixed#2": 0.8, "Fixed#3": 0.1}  # each member's own score
    assert res["violence"].raw["Fixed#2"].flagged is True
    assert res.source == "EnsembleGuard"


async def test_ensemble_members_run_concurrently():
    members = [Fixed({"violence": 0.9}, delay=0.1) for _ in range(3)]
    g = EnsembleGuard(members, rule="any")
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await g.guard("x")
    assert loop.time() - t0 < 0.25  # three 0.1 s members, not 0.3 s
    assert all(m.calls == 1 for m in members)


async def test_ensemble_works_with_guard_many_and_rethresholding():
    g = EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.2})], rule="all")
    results = await g.guard_many(["a", "b", "c"])
    assert len(results) == 3 and not results[0].flagged
    assert results[0].at(0.5).flagged  # half the members flag: passes a 0.5 cut-off
    assert isinstance(g, Guard)


def test_ensemble_validates_its_members():
    with pytest.raises(ValueError, match="threshold"):
        EnsembleGuard([Fixed({"violence": 0.9}, threshold=None), Fixed({"violence": 0.9})])
    with pytest.raises(ValueError, match="same categories"):
        EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"pii": 0.9})])
    with pytest.raises(ValueError, match="k"):
        EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.9})], rule="votes", k=3)
    with pytest.raises(ValueError, match="rule"):
        EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.9})], rule="mean")
    with pytest.raises(ValueError, match="at least two"):
        EnsembleGuard([Fixed({"violence": 0.9})])


# ── CascadeGuard ───────────────────────────────────────────────────────────
async def test_cascade_falls_back_on_error():
    broken = Fixed({"violence": 0.9}, error=RuntimeError("502 Bad Gateway"))
    backup = Fixed({"violence": 0.3})
    res = await CascadeGuard([broken, backup]).guard("x")
    assert res["violence"].score == pytest.approx(0.3)
    assert res.source == "Fixed#2"
    assert broken.calls == 1 and backup.calls == 1


async def test_cascade_raises_when_every_guard_fails():
    a = Fixed({"violence": 0.9}, error=RuntimeError("down"))
    b = Fixed({"violence": 0.9}, error=TimeoutError("slow"))
    with pytest.raises(TimeoutError, match="slow"):
        await CascadeGuard([a, b]).guard("x")


async def test_cascade_escalates_only_when_unsure():
    sure = Fixed({"violence": 0.05, "pii": 0.95})
    judge = Fixed({"violence": 0.5, "pii": 0.5})
    res = await CascadeGuard([sure, judge], uncertain=(0.2, 0.8)).guard("x")
    assert res.source == "Fixed" and judge.calls == 0  # every score outside the band: the first guard answers

    unsure = Fixed({"violence": 0.45, "pii": 0.95})
    res = await CascadeGuard([unsure, judge], uncertain=(0.2, 0.8)).guard("x")
    assert res.source == "Fixed#2" and judge.calls == 1  # one score in the band: escalate


async def test_cascade_without_a_band_never_escalates_a_good_answer():
    first, second = Fixed({"violence": 0.5}), Fixed({"violence": 0.9})
    res = await CascadeGuard([first, second]).guard("x")
    assert res.source == "Fixed" and second.calls == 0


async def test_cascade_last_guard_answers_even_if_unsure():
    a, b = Fixed({"violence": 0.5}), Fixed({"violence": 0.5})
    res = await CascadeGuard([a, b], uncertain=(0.2, 0.8)).guard("x")
    assert res.source == "Fixed#2"


async def test_cascade_passes_input_too_long_to_the_next_guard():
    short = Fixed({"violence": 0.9}, error=InputTooLong("too long"))
    long = Fixed({"violence": 0.1})
    res = await CascadeGuard([short, long]).guard("x")
    assert res.source == "Fixed#2"


def test_cascade_validates():
    with pytest.raises(ValueError, match="at least two"):
        CascadeGuard([Fixed({"violence": 0.9})])
    with pytest.raises(ValueError, match="uncertain"):
        CascadeGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.9})], uncertain=(0.8, 0.2))
    with pytest.raises(ValueError, match="same categories"):
        CascadeGuard([Fixed({"violence": 0.9}), Fixed({"pii": 0.9})])


async def test_composites_nest():
    ens = EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.8})], rule="all")
    broken = Fixed({"violence": 0.9}, error=RuntimeError("down"))
    res = await CascadeGuard([broken, ens]).guard("x")
    assert res.source == "EnsembleGuard" and res.flagged


# ── rules ──────────────────────────────────────────────────────────────────
from reflexguard.compose import All, Any, Rule, Votes, Weighted  # noqa: E402


def three():
    return [Fixed({"violence": 0.9, "pii": 0.1}), Fixed({"violence": 0.8, "pii": 0.2}), Fixed({"violence": 0.1, "pii": 0.7})]


@pytest.mark.parametrize("rule, flagged", [(Any(), {"violence", "pii"}), (All(), set()), (Votes(2), {"violence"}),
                                           (Votes(), {"violence"})])
async def test_built_in_rules_match_the_string_shortcuts(rule, flagged):
    res = await EnsembleGuard(three(), rule=rule).guard("x")
    assert {c.name for c in res.violations} == flagged


async def test_weighted_rule():
    # the third member alone carries more weight than the first two together
    rule = Weighted({"Fixed": 1, "Fixed#2": 1, "Fixed#3": 3}, cutoff=0.5)
    res = await EnsembleGuard(three(), rule=rule).guard("x")
    assert {c.name for c in res.violations} == {"pii"}
    assert res["pii"].score == pytest.approx(3 / 5)


async def test_a_custom_rule_is_any_object_with_score_and_threshold():
    class FirstMustAgree:
        """Flag only when the first member flags and at least one other does."""

        def score(self, members):
            first, *rest = members.values()
            return float(bool(first.flagged) and any(m.flagged for m in rest))

        def threshold(self, n):
            return 1.0

    assert isinstance(FirstMustAgree(), Rule)
    res = await EnsembleGuard(three(), rule=FirstMustAgree()).guard("x")
    assert {c.name for c in res.violations} == {"violence"}


def test_rule_validation():
    with pytest.raises(ValueError, match="k"):
        EnsembleGuard(three(), rule=Votes(4))
    with pytest.raises(ValueError, match="k only applies"):
        EnsembleGuard(three(), rule=Any(), k=2)
    with pytest.raises(ValueError, match="weights"):
        EnsembleGuard(three(), rule=Weighted({"Fixed": 1}))
