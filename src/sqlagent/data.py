import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPIDER_DIR = Path("data/spider")

SPLIT_FILES = {
    "train": "train_spider.json",
    "dev": "dev.json",
}


@dataclass(frozen=True)
class Example:
    question: str
    db_id: str
    gold_sql: str
    db_path: Path


def load_split(split: str, spider_dir: Path = DEFAULT_SPIDER_DIR) -> list[Example]:
    """Load a Spider split ('train' or 'dev') as a list of Examples."""
    if split not in SPLIT_FILES:
        raise ValueError(f"Unknown split {split!r}; expected one of {list(SPLIT_FILES)}")

    spider_dir = Path(spider_dir)
    with open(spider_dir / SPLIT_FILES[split], encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for item in raw:
        db_id = item["db_id"]
        db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"Missing database file: {db_path}")
        examples.append(
            Example(
                question=item["question"],
                db_id=db_id,
                gold_sql=item["query"],
                db_path=db_path,
            )
        )
    return examples