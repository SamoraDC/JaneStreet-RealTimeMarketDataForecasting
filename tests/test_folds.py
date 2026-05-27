import pytest

from janestreet.folds import make_expanding_folds, make_rolling_folds


def test_make_expanding_folds_uses_recent_validation_blocks():
    folds = make_expanding_folds(
        min_date_id=0,
        max_date_id=99,
        n_folds=3,
        valid_window=10,
        gap=2,
        min_train_window=50,
    )

    assert [fold.name for fold in folds] == ["wf_01", "wf_02", "wf_03"]
    assert [(fold.train_start, fold.train_end, fold.valid_start, fold.valid_end) for fold in folds] == [
        (0, 67, 70, 79),
        (0, 77, 80, 89),
        (0, 87, 90, 99),
    ]


def test_make_expanding_folds_rejects_impossible_windows():
    with pytest.raises(ValueError, match="not enough dates"):
        make_expanding_folds(
            min_date_id=0,
            max_date_id=20,
            n_folds=3,
            valid_window=10,
        )

    with pytest.raises(ValueError, match="minimum"):
        make_expanding_folds(
            min_date_id=0,
            max_date_id=99,
            n_folds=3,
            valid_window=10,
            min_train_window=80,
        )


def test_make_rolling_folds_keeps_fixed_train_window():
    folds = make_rolling_folds(
        min_date_id=0,
        max_date_id=99,
        n_folds=2,
        train_window=30,
        valid_window=10,
        gap=1,
    )

    assert [(fold.train_start, fold.train_end, fold.valid_start, fold.valid_end) for fold in folds] == [
        (49, 78, 80, 89),
        (59, 88, 90, 99),
    ]

