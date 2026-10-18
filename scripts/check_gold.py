from tqdm import tqdm

from sqlagent.data import load_split
from sqlagent.reward import execute_sql, execution_reward

for split in ["dev", "train"]:
    examples = load_split(split)
    failed, empty, not_one = 0, 0, 0
    for ex in tqdm(examples, desc=split):
        res = execute_sql(ex.db_path, ex.gold_sql)
        if not res.ok:
            failed += 1
            continue
        if not res.rows:
            empty += 1
        if execution_reward(ex.gold_sql, ex.gold_sql, ex.db_path) != 1.0:
            not_one += 1
    print(f"{split}: {len(examples)} examples, {failed} gold failures, "
          f"{empty} empty gold results, {not_one} gold-vs-gold mismatches")