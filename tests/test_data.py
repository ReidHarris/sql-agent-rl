from pathlib import Path

import pytest

from sqlagent.data import DEFAULT_SPIDER_DIR, load_split

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_SPIDER_DIR).exists(),
    reason="Spider data not downloaded",
)


def test_dev_split_size():
    examples = load_split("dev")
    assert len(examples) == 1034


def test_examples_are_well_formed():
    for ex in load_split("dev")[:50]:
        assert ex.question
        assert ex.gold_sql
        assert ex.db_path.exists()


def test_unknown_split_raises():
    with pytest.raises(ValueError):
        load_split("bogus")