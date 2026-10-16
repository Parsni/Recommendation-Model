"""Deterministic generator for the synthetic demo dataset.

Run:  python data/synthetic_demo/generate_demo_data.py

The dataset is domain-neutral placeholder data used ONLY to verify engine
functionality. Its known expected outcomes are documented in KNOWN_FACTS.md
in this folder. Regenerating always produces byte-identical CSV files.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
START = datetime(2025, 1, 1)

GROUPS = {
    "A": {"users": "alpha", "items": ["I001", "I002", "I003", "I004"], "category": "cat_alpha"},
    "B": {"users": "beta", "items": ["I005", "I006", "I007", "I008"], "category": "cat_beta"},
    "C": {"users": "gamma", "items": ["I009", "I010", "I011", "I012"], "category": "cat_gamma"},
}
SHARED_ITEM = "I013"
UNSEEN_ITEMS = [f"I{n:03d}" for n in range(14, 21)]
REGIONS = ["north", "south", "east", "west"]

DESC_POOLS = {
    "A": ["compact solar garden light weatherproof outdoor lantern patio illumination",
          "rechargeable led garden lamp solar powered outdoor lighting path glow",
          "solar patio lantern weatherproof garden lighting outdoor dusk illumination",
          "outdoor solar pathway light weather resistant garden lamp evening glow"],
    "B": ["stainless coffee brewing kettle temperature control pour over caffeine",
          "gooseneck kettle precision temperature barista coffee pour caffeine brew",
          "thermal carafe coffee maker insulated brew caffeine morning preparation",
          "espresso moka pot stovetop coffee brewing rich caffeine preparation"],
    "C": ["lightweight running shoe cushioned marathon training athletic jog",
          "breathable road running sneakers foam cushion marathon athletic training",
          "trail running shoes gripped sole marathon training athletic jogging",
          "cushioned sprint running flats lightweight tempo training athletic shoe"],
}


def user_group(u: int) -> str:
    return ["A", "B", "C"][(u - 1) % 3]


def rating(u: int, item_num: int) -> int:
    return 3 + ((u + item_num) % 3)


def build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for u in range(1, 31):
        grp = GROUPS[user_group(u)]
        order = [SHARED_ITEM] + list(reversed(grp["items"]))  # last interaction = first group item
        for k, item in enumerate(order):
            ts = START + timedelta(days=(u - 1) * 5 + k)
            rows.append({
                "user_id": f"U{u:03d}",
                "item_id": item,
                "interaction_value": rating(u, int(item[1:])),
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
    interactions = pd.DataFrame(rows)

    # --- seeded, documented data-quality defects (see KNOWN_FACTS.md) ---
    dup1 = interactions.iloc[3].to_dict()      # exact duplicate of row 3
    dup2 = interactions.iloc[40].to_dict()     # exact duplicate of row 40
    bad_id = {"user_id": "", "item_id": "I002", "interaction_value": 4,
              "timestamp": "2025-02-01 00:00:00"}                      # missing identifier
    bad_val = {"user_id": "U007", "item_id": "I013", "interaction_value": "great",
               "timestamp": "2025-03-01 00:00:00"}                      # non-numeric value
    interactions = pd.concat([interactions,
                              pd.DataFrame([dup1, dup2, bad_id, bad_val])], ignore_index=True)

    item_rows = []
    for g, spec in GROUPS.items():
        for j, item in enumerate(spec["items"]):
            item_rows.append({"item_id": item, "category": spec["category"],
                              "description": DESC_POOLS[g][j],
                              "price": 10 + int(item[1:]) * 7})
    item_rows.append({"item_id": SHARED_ITEM, "category": "cat_common",
                      "description": "universal power adapter fast charging cable compatible accessory device",
                      "price": 10 + 13 * 7})
    for item in UNSEEN_ITEMS:
        item_rows.append({"item_id": item, "category": "cat_misc",
                          "description": f"uncategorized spare product number {item[1:]} no history yet",
                          "price": 10 + int(item[1:]) * 7})
    items = pd.DataFrame(item_rows)

    user_rows = []
    for u in range(1, 31):
        age = 22 + (u * 7) % 40
        band = "young" if age <= 30 else ("middle" if age <= 45 else "senior")
        user_rows.append({
            "user_id": f"U{u:03d}", "age": age, "age_band": band,
            "region": REGIONS[u % 4], "preferred_category": GROUPS[user_group(u)]["category"],
        })
    users = pd.DataFrame(user_rows)
    return interactions, items, users


if __name__ == "__main__":
    inter, items, users = build()
    inter.to_csv(OUT / "interactions.csv", index=False)
    items.to_csv(OUT / "items.csv", index=False)
    users.to_csv(OUT / "users.csv", index=False)
    print(f"interactions.csv rows written: {len(inter)}")
    print(f"items.csv rows written: {len(items)}")
    print(f"users.csv rows written: {len(users)}")
