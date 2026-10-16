import pytest

from reflexguard import Guard
from reflexguard.questions import SCORE_LEVELS


class FakeGuard(Guard):
    """Answers every question from a {(mode, category): value} table."""

    def __init__(self, table, **kwargs):
        super().__init__(**kwargs)
        self.table = table
        self.calls = []

    async def predict(self, context, questions):
        self.calls.append((context, questions))
        answers = {}
        for qid in questions:
            mode, cat = qid.split(":", 1)
            v = self.table[(mode, cat)]
            if mode == "noul":
                answers[qid] = {"noul": v}
            elif mode == "choice":
                answers[qid] = {"probabilities": {cat: v, f"not_{cat}": 1 - v}}
            else:
                answers[qid] = {"score": v * (len(SCORE_LEVELS) - 1)}
        return answers


TABLE = {
    ("noul", "violence"): 0.9, ("choice", "violence"): 0.7, ("score", "violence"): 0.2,
    ("noul", "pii"): 0.1, ("choice", "pii"): 0.6, ("score", "pii"): 0.5,
}  # fmt: skip
N, C, S = Guard.NOUL, Guard.CHOICE, Guard.SCORE


def fake(**kwargs):
    kwargs.setdefault("categories", ["violence", "pii"])
    return FakeGuard(TABLE, **kwargs)


def test_is_a_guard():
    assert isinstance(fake(), Guard)
    assert not isinstance(object(), Guard)


async def test_default_mode_is_noul_and_ranks():
    g = fake()
    res = await g.guard("hello")
    assert list(g.calls[0][1]) == ["noul:violence", "noul:pii"]
    assert [c.name for c in res.ranked] == ["violence", "pii"]
    assert res["violence"].by_mode == {"noul": 0.9}
    assert res.flagged is None


async def test_one_predict_call_for_all_modes():
    g = fake(mode=N | C | S)
    await g.guard("hello")
    assert len(g.calls) == 1
    assert len(g.calls[0][1]) == 6


async def test_or_and_votes():
    assert (await fake(mode=N | C | S).guard("x")).scores == {"violence": 0.9, "pii": 0.6}
    assert (await fake(mode=N & C & S).guard("x"))["violence"].score == pytest.approx(0.2)
    res = await fake(mode=N | C | S, votes=2).guard("x")
    assert res["violence"].score == pytest.approx(0.7)
    assert res["pii"].score == pytest.approx(0.5)


async def test_threshold_gives_violations():
    res = await fake(mode=N | C, votes=2, threshold=0.65).guard("x")
    assert [c.name for c in res.violations] == ["violence"]
    assert res.flagged is True


async def test_raw_answers_kept():
    res = await fake(mode=N | C).guard("x")
    assert set(res["pii"].raw) == {"noul:pii", "choice:pii"}
    assert len(res.raw) == 4


def test_dict_categories_pass_descriptions():
    g = fake(categories={"violence": "threats", "pii": ""})
    g.guard_sync("x")
    q = g.calls[0][1]["noul:violence"]
    assert q["instructions"] == "Does `text` involve violence (threats)?"


def test_sync_guard_outside_loop():
    assert fake().guard_sync("x").top.name == "violence"


async def test_sync_guard_inside_running_loop():
    # e.g. called from a Jupyter / marimo cell
    assert fake().guard_sync("x").top.name == "violence"


async def test_missing_answer_raises():
    class Broken(FakeGuard):
        async def predict(self, context, questions):
            return {}

    with pytest.raises(KeyError, match="noul:violence"):
        await Broken(TABLE, categories=["violence"]).guard("x")


@pytest.mark.parametrize("ctx, err", [("", ValueError), ("   ", ValueError), (None, TypeError)])
async def test_bad_context(ctx, err):
    with pytest.raises(err):
        await fake().guard(ctx)


@pytest.mark.parametrize(
    "kwargs, err",
    [
        ({"categories": "violence"}, TypeError),  # a bare string would split into letters
        ({"categories": []}, ValueError),
        ({"categories": [""]}, ValueError),
        ({"categories": {"a": None}}, TypeError),
        ({"mode": "noul"}, TypeError),
        ({"mode": N, "votes": 1}, ValueError),
        ({"threshold": 2}, ValueError),
        ({"threshold": {"violence": 0.5}}, ValueError),
    ],
)
def test_bad_construction(kwargs, err):
    with pytest.raises(err):
        fake(**kwargs)


async def test_bare_guard_has_no_model():
    with pytest.raises(NotImplementedError, match="must implement predict"):
        await Guard(categories=["a"]).guard("x")


def test_debug_is_off_by_default(logs):
    g = fake()
    assert g.debug is False
    g.guard_sync("x")
    assert logs == []


def test_debug_logs_questions_timing_and_scores(logs):
    fake(mode=N | C, threshold=0.5, debug=True).guard_sync("x")
    text = "\n".join(logs)
    assert "FakeGuard: 4 questions for 2 categories" in text
    assert "predict took" in text
    assert "violence score=0.900" in text
    assert "flagged=True" in text


def test_version_matches_the_package_metadata():
    from importlib.metadata import version

    import reflexguard

    assert reflexguard.__version__ == version("reflexguard") == "0.2.1"


class ChunkedGuard(FakeGuard):
    """Answers one table per chunk, as a guard does for long text."""

    def __init__(self, tables, **kwargs):
        super().__init__(tables[0], **kwargs)
        self.tables = tables

    async def predict(self, context, questions):
        out = []
        for t in self.tables:
            self.table = t
            out.append(await super().predict(context, questions))
        return out


def chunk(**values):
    """chunk(noul_violence=0.9, ...) -> a full table, 0.0 where not given."""
    t = dict.fromkeys(TABLE, 0.0)
    for k, v in values.items():
        mode, cat = k.split("_", 1)
        t[(mode, cat)] = v
    return t


async def test_modes_must_agree_within_one_chunk():
    # noul is high only in chunk 0, choice only in chunk 1: nowhere do both agree
    g = ChunkedGuard([chunk(noul_violence=0.9, choice_violence=0.1), chunk(noul_violence=0.1, choice_violence=0.9)],
                     categories=["violence"], mode=N & C)
    res = await g.guard("long text")
    assert res["violence"].score == pytest.approx(0.1)
    assert res.chunks == 2


async def test_the_chunk_that_sets_the_score_supplies_by_mode_and_raw():
    g = ChunkedGuard([chunk(noul_violence=0.3, choice_violence=0.2), chunk(noul_violence=0.8, choice_violence=0.7)],
                     categories=["violence"], mode=N | C)
    res = await g.guard("long text")
    v = res["violence"]
    assert v.chunk == 1
    assert v.by_mode == pytest.approx({"noul": 0.8, "choice": 0.7})
    assert v.raw["noul:violence"] == {"noul": 0.8}


async def test_categories_are_judged_by_their_own_worst_chunk():
    # multi-label: violence is in chunk 0, pii in chunk 2, and both are flagged
    g = ChunkedGuard([chunk(noul_violence=0.9), chunk(), chunk(noul_pii=0.8)], categories=["violence", "pii"],
                     threshold=0.5)
    res = await g.guard("long text")
    assert {c.name for c in res.violations} == {"violence", "pii"}
    assert (res["violence"].chunk, res["pii"].chunk) == (0, 2)


async def test_one_chunk_keeps_the_old_shape():
    res = await fake().guard("hello")
    assert res.chunks == 1 and res["violence"].chunk == 0


async def test_guard_many_keeps_order_and_limits_concurrency():
    import asyncio

    class Slow(FakeGuard):
        active = peak = 0

        async def predict(self, context, questions):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            await asyncio.sleep(0.01)
            Slow.active -= 1
            return await super().predict(context, questions)

    g = Slow(TABLE, categories=["violence", "pii"])
    texts = [f"text {i}" for i in range(7)]
    results = await g.guard_many(texts, concurrency=3)
    assert [ctx for ctx, _ in g.calls] and len(results) == 7
    assert Slow.peak == 3
    assert all(r.top.name == "violence" for r in results)
    Slow.peak = 0
    await g.guard_many(texts)  # the guard's default
    assert Slow.peak == Guard.concurrency == 2


async def test_guard_many_rejects_zero_concurrency():
    with pytest.raises(ValueError, match="concurrency"):
        await fake().guard_many(["a"], concurrency=0)
