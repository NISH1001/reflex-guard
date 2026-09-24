import pytest

from reflexguard.modes import AllOf, AnyOf, Mode, with_votes

N, C, S = Mode.NOUL, Mode.CHOICE, Mode.SCORE
SCORES = {"noul": 0.9, "choice": 0.6, "score": 0.2}


def test_single_mode_returns_its_score():
    assert N.combine(SCORES) == 0.9
    assert N.modes() == (N,)


def test_or_flattens_and_takes_max():
    expr = N | C | S
    assert expr == AnyOf((N, C, S))
    assert expr.combine(SCORES) == 0.9


def test_and_flattens_and_takes_min():
    expr = N & C & S
    assert expr == AllOf((N, C, S))
    assert expr.combine(SCORES) == 0.2


def test_nested_expression():
    expr = (N & S) | C
    assert expr.combine(SCORES) == 0.6  # max(min(0.9, 0.2), 0.6)
    assert expr.modes() == (N, S, C)


def test_votes_takes_kth_highest():
    assert with_votes(N | C | S, 2).combine(SCORES) == 0.6
    assert with_votes(N | C | S, 1).combine(SCORES) == 0.9
    assert with_votes(N | C | S, 3).combine(SCORES) == 0.2


@pytest.mark.parametrize("votes", [0, 4, -1, 1.5, True])
def test_votes_out_of_range(votes):
    with pytest.raises(ValueError, match="votes must be"):
        with_votes(N | C | S, votes)


@pytest.mark.parametrize("expr", [N, N & C])
def test_votes_needs_or_group(expr):
    with pytest.raises(ValueError, match="plain `|` group"):
        with_votes(expr, 1)


def test_duplicate_modes_are_listed_once():
    assert (N | N).modes() == (N,)
    assert with_votes(N | N, 2).combine(SCORES) == 0.9


def test_combining_with_non_mode_raises():
    with pytest.raises(TypeError, match="only combine modes"):
        N | "choice"
