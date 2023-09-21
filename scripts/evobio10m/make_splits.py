"""
Decides which images are part of the validation split
"""

import argparse
import os.path
import random

from tqdm import tqdm

from imageomics import evobio10m

insert_stmt = (
    "INSERT INTO split (evobio10m_id, is_val, is_train_small) VALUES (?, ?, ?)"
)


def make_split(table_name):
    select_stmt = f"SELECT evobio10m_id FROM {table_name} ORDER BY evobio10m_id"
    all_ids = set([row[0] for row in db.execute(select_stmt).fetchall()])

    random.seed(args.seed)
    val_ids = set(random.sample(list(all_ids), k=len(all_ids) * args.val_split // 100))

    train_ids = all_ids - val_ids

    train_small_ids = set(
        random.sample(list(train_ids), k=len(train_ids) * args.train_small_split // 100)
    )

    for id in tqdm(all_ids, desc=table_name):
        db.execute(insert_stmt, (id, id in val_ids, id in train_small_ids))
    db.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to mapping.sqlite")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument(
        "--val-split",
        type=int,
        default=5,
        help="Percentage of all data to use for validation",
    )
    parser.add_argument(
        "--train-small-split",
        type=int,
        default=5,
        help="Percentage of training data to use for ablation",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.db)
    db = evobio10m.get_db(args.db)

    make_split("eol")
    make_split("inat21")
    make_split("bioscan")
