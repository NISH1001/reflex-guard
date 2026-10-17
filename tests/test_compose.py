import asyncio

import pytest

from reflexguard import (AllOf, AnyOf, CascadeGuard, EnsembleGuard, EnsembleRule, EscalationRule, Guard, InputTooLong,
                         Margin, Rule, Votes, Weighted)


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
async def test_ensemble_score_is_the_share_of_members_flagging():
    a = Fixed({"violence": 0.9, "pii": 0.1})
    b = Fixed({"violence": 0.8, "pii": 0.2})
    c = Fixed({"violence": 0.1, "pii": 0.7})
    res = await EnsembleGuard([a, b, c], rule=Votes(2)).guard("x")
    assert res["violence"].score == pytest.approx(2 / 3) and res["violence"].threshold == pytest.approx(2 / 3)
    assert res["pii"].score == pytest.approx(1 / 3)
    assert res["violence"].by_mode == {"Fixed": 0.9, "Fixed#2": 0.8, "Fixed#3": 0.1}  # each member's own score
    assert res["violence"].raw["Fixed#2"].flagged is True
    assert res.source == "EnsembleGuard"


async def test_ensemble_members_run_concurrently():
    members = [Fixed({"violence": 0.9}, delay=0.1) for _ in range(3)]
    g = EnsembleGuard(members, rule=AnyOf())
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await g.guard("x")
    assert loop.time() - t0 < 0.25  # three 0.1 s members, not 0.3 s
    assert all(m.calls == 1 for m in members)


async def test_ensemble_parallel_caps_members_in_flight():
    members = [Fixed({"violence": 0.9}, delay=0.1) for _ in range(4)]
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await EnsembleGuard(members, rule=AnyOf(), parallel=2).guard("x")
    assert 0.18 < loop.time() - t0 < 0.3  # two rounds of two


def test_ensemble_parallel_bounds():
    two = [Fixed({"violence": 0.9}), Fixed({"violence": 0.9})]
    assert EnsembleGuard(two).parallel == 2
    with pytest.raises(ValueError, match="parallel"):
        EnsembleGuard(two, parallel=0)
    from loguru import logger

    seen = []
    sink = logger.add(seen.append, level="WARNING")
    try:
        assert EnsembleGuard(two, parallel=5).parallel == 2
    finally:
        logger.remove(sink)
    assert any("parallel=5" in m for m in seen)


async def test_ensemble_works_with_guard_many_and_rethresholding():
    g = EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.2})], rule=AllOf())
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
        EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.9})], rule=Votes(3))
    with pytest.raises(TypeError, match="EnsembleRule"):
        EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.9})], rule="votes")
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


async def test_cascade_escalates_near_the_first_guards_threshold():
    judge = Fixed({"violence": 0.5, "pii": 0.5})
    sure = Fixed({"violence": 0.05, "pii": 0.95})  # top score 0.95, far above 0.5
    res = await CascadeGuard([sure, judge], rule=Margin(0.2)).guard("x")
    assert res.source == "Fixed" and judge.calls == 0

    unsure = Fixed({"violence": 0.05, "pii": 0.6})  # top score within 0.2 of 0.5
    res = await CascadeGuard([unsure, judge], rule=Margin(0.2)).guard("x")
    assert res.source == "Fixed#2" and judge.calls == 1


@pytest.mark.parametrize("top, escalates", [(0.25, True), (0.15, False), (0.55, True), (0.65, False)])
async def test_margin_can_be_asymmetric(top, escalates):
    rule = Margin(0.3, 0.1)  # [0.2, 0.6) around a 0.5 threshold
    res = await Fixed({"violence": top, "pii": 0.0}).guard("x")
    assert rule.escalate(res) is escalates


async def test_margin_is_relative_to_each_categorys_threshold():
    g = Fixed({"violence": 0.75, "pii": 0.3}, threshold={"violence": 0.7, "pii": 0.9})
    res = await g.guard("x")  # violence flagged by 0.05: near its own threshold
    assert Margin(0.1).escalate(res) and not Margin(0.01).escalate(res)


async def test_cascade_without_a_rule_never_escalates_a_good_answer():
    first, second = Fixed({"violence": 0.5}), Fixed({"violence": 0.9})
    res = await CascadeGuard([first, second]).guard("x")
    assert res.source == "Fixed" and second.calls == 0


async def test_cascade_last_guard_answers_even_if_unsure():
    a, b = Fixed({"violence": 0.5}), Fixed({"violence": 0.5})
    res = await CascadeGuard([a, b], rule=Margin(0.2)).guard("x")
    assert res.source == "Fixed#2"


async def test_cascade_takes_one_rule_per_handoff():
    a, b, c = Fixed({"violence": 0.52}), Fixed({"violence": 0.6}), Fixed({"violence": 0.9})
    res = await CascadeGuard([a, b, c], rule=[Margin(0.05), Margin(0.05)]).guard("x")
    assert res.source == "Fixed#2" and c.calls == 0  # a escalates (0.02 away), b is sure enough (0.1 away)
    res = await CascadeGuard([a, b, c], rule=[Margin(0.05), Margin(0.2)]).guard("x")
    assert res.source == "Fixed#3"


async def test_a_custom_escalation_rule():
    class DoubleCheckFlags:
        def escalate(self, result):
            return bool(result.flagged)

    assert isinstance(DoubleCheckFlags(), EscalationRule) and not isinstance(DoubleCheckFlags(), EnsembleRule)
    a, b = Fixed({"violence": 0.9}), Fixed({"violence": 0.1})
    res = await CascadeGuard([a, b], rule=DoubleCheckFlags()).guard("x")
    assert res.source == "Fixed#2" and not res.flagged


async def test_cascade_passes_input_too_long_to_the_next_guard():
    short = Fixed({"violence": 0.9}, error=InputTooLong("too long"))
    long = Fixed({"violence": 0.1})
    res = await CascadeGuard([short, long]).guard("x")
    assert res.source == "Fixed#2"


def test_cascade_validates():
    with pytest.raises(ValueError, match="at least two"):
        CascadeGuard([Fixed({"violence": 0.9})])
    two = [Fixed({"violence": 0.9}), Fixed({"violence": 0.9})]
    with pytest.raises(TypeError, match="EscalationRule"):
        CascadeGuard(two, rule=Votes(2))
    with pytest.raises(TypeError, match="EnsembleRule"):
        EnsembleGuard(two, rule=Margin(0.2))
    with pytest.raises(ValueError, match="per handoff"):
        CascadeGuard(two + [Fixed({"violence": 0.9})], rule=[Margin(0.2)])
    with pytest.raises(ValueError, match="threshold"):
        CascadeGuard([Fixed({"violence": 0.9}, threshold=None), Fixed({"violence": 0.9})], rule=Margin(0.2))
    with pytest.raises(ValueError, match=">= 0"):
        Margin(-0.1)
    with pytest.raises(ValueError, match="same categories"):
        CascadeGuard([Fixed({"violence": 0.9}), Fixed({"pii": 0.9})])


async def test_composites_nest():
    ens = EnsembleGuard([Fixed({"violence": 0.9}), Fixed({"violence": 0.8})], rule=AllOf())
    broken = Fixed({"violence": 0.9}, error=RuntimeError("down"))
    res = await CascadeGuard([broken, ens]).guard("x")
    assert res.source == "EnsembleGuard" and res.flagged


# ── rules ──────────────────────────────────────────────────────────────────


def three():
    return [Fixed({"violence": 0.9, "pii": 0.1}), Fixed({"violence": 0.8, "pii": 0.2}), Fixed({"violence": 0.1, "pii": 0.7})]


@pytest.mark.parametrize("rule, flagged", [(AnyOf(), {"violence", "pii"}), (AllOf(), set()), (Votes(2), {"violence"}),
                                           (Votes(), {"violence"})])
async def test_built_in_rules(rule, flagged):
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

    assert isinstance(FirstMustAgree(), EnsembleRule)
    res = await EnsembleGuard(three(), rule=FirstMustAgree()).guard("x")
    assert {c.name for c in res.violations} == {"violence"}


def test_rule_validation():
    with pytest.raises(ValueError, match="k"):
        EnsembleGuard(three(), rule=Votes(4))
    with pytest.raises(ValueError, match="weights"):
        EnsembleGuard(three(), rule=Weighted({"Fixed": 1}))


# ── operators ──────────────────────────────────────────────────────────────
def test_or_builds_an_anyof_ensemble_and_and_an_allof_one():
    a, b = Fixed({"violence": 0.9}), Fixed({"violence": 0.2})
    g = a | b
    assert isinstance(g, EnsembleGuard) and isinstance(g.rule, AnyOf) and g.guards == [a, b]
    g = a & b
    assert isinstance(g, EnsembleGuard) and isinstance(g.rule, AllOf) and g.guards == [a, b]


def test_chains_flatten_and_mixes_nest():
    a, b, c = Fixed({"violence": 0.9}), Fixed({"violence": 0.2}), Fixed({"violence": 0.6})
    assert (a | b | c).guards == [a, b, c] and (a & b & c).guards == [a, b, c]
    mixed = a | (b & c)
    assert isinstance(mixed.rule, AnyOf) and mixed.guards[0] is a
    inner = mixed.guards[1]
    assert isinstance(inner.rule, AllOf) and inner.guards == [b, c]
    assert isinstance((a | b) & c, EnsembleGuard) and ((a | b) & c).guards[1] is c  # & does not flatten an AnyOf


async def test_operator_ensembles_decide_like_their_rules():
    a, b = Fixed({"violence": 0.9}), Fixed({"violence": 0.2})
    assert (await (a | b).guard("x")).flagged is True
    assert (await (a & b).guard("x")).flagged is False


def test_operators_reject_non_guards_and_members_without_thresholds():
    a = Fixed({"violence": 0.9})
    with pytest.raises(TypeError):
        a | 3
    with pytest.raises(ValueError, match="threshold"):
        a | Fixed({"violence": 0.9}, threshold=None)


def test_rshift_builds_a_cascade_and_chains_flatten():
    a, b, c = Fixed({"violence": 0.9}), Fixed({"violence": 0.2}), Fixed({"violence": 0.6})
    g = a >> b >> c
    assert isinstance(g, CascadeGuard) and g.guards == [a, b, c] and g.rules == [None, None]
    tuned = CascadeGuard([a, b], rule=Margin(0.2))
    assert (tuned >> c).guards == [tuned, c]  # a cascade with rules stays whole
    mixed = a >> (b & c)
    assert isinstance(mixed.guards[1], EnsembleGuard)
    with pytest.raises(TypeError):
        a >> 3


def test_built_in_rules_are_rules_of_their_kind():
    for r in (Votes(2), AnyOf(), AllOf(), Weighted({"a": 1})):
        assert isinstance(r, Rule) and isinstance(r, EnsembleRule) and not isinstance(r, EscalationRule)
    assert isinstance(Margin(0.2), Rule) and isinstance(Margin(0.2), EscalationRule)
    assert not isinstance(Margin(0.2), EnsembleRule)


def test_modes_still_combine_with_or():
    assert Guard.NOUL | Guard.CHOICE  # the class-level Mode algebra is untouched
