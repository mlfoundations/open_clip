"""
Decides which images are part of the validation split
"""

import argparse
import os.path
import random

from tqdm import tqdm

from imageomics import evobio10m


def make_split(table_name):
    select_stmt = f"SELECT evobio10m_id FROM {table_name} ORDER BY evobio10m_id"
    all_ids = [row[0] for row in db.execute(select_stmt).fetchall()]

    random.seed(args.seed)
    val_ids = set(random.sample(all_ids, k=len(all_ids) * args.val_split // 100))

    insert_stmt = "INSERT INTO split (evobio10m_id, is_train) VALUES (?, ?)"
    for id in tqdm(all_ids, desc=table_name):
        db.execute(insert_stmt, (id, id not in val_ids))
    db.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to mapping.sqlite")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument(
        "--val-split",
        type=int,
        default=5,
        help="Percentage of training data to use for validation",
    )
    args = parser.parse_args()

    assert os.path.isfile(args.db)
    db = evobio10m.get_db(args.db)

    make_split("eol")
    make_split("bioscan")
    make_split("inat21")
