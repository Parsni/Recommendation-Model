# Data schema documentation

## Generic contract

Column names are **configurable** in `config/default_config.json` under
`column_mapping`; canonical semantics below are what the engine needs.

### Interactions (required file)

| canonical column | required | type | notes |
|---|---|---|---|
| user_id | yes | string | blank ids → row dropped & recorded |
| item_id | yes | string | blank ids → row dropped & recorded |
| interaction_value | no | numeric | non-numeric coerced to NaN & counted; absence means "every interaction is a positive event" |
| timestamp | no | datetime-parseable | unparseable → NaT & recorded; absence disables chronological splitting & time-series stats |
| anything else | no | - | kept and surfaced as metadata candidates, never silently removed |

### Items (optional)
`item_id` plus any categorical / numerical / free-text description columns.
Field roles are read from config when given, otherwise inferred
(distinct-count and length heuristics in `features/preprocess.py`).

### Users (optional)
`user_id` plus categorical / numerical attributes.

## Validation behaviour (src/data/schema.py)

For every load the engine reports, per file:

- rows read vs usable;
- missing required columns (hard stop for that algorithm family);
- dropped malformed rows (indices recorded, capped at 1000);
- exact duplicate counts and post-dedup behaviour
  (`preprocessing.deduplicate_key`, default user+item+timestamp);
- missing-value counts per column;
- dtype coercion issues (values, timestamps);
- warnings (e.g. "no timestamp → chronological evaluation unavailable").

Original files are never modified; all cleaning happens on copies.
Non-CSV uploads are rejected with a clear message.

## Synthetic demo schema (data/synthetic_demo/)

- `interactions.csv`: 154 rows including 4 deliberately seeded defects
  (documented in `KNOWN_FACTS.md`: 1 blank user_id, 2 exact duplicates,
  1 non-numeric value) → validation must catch exactly these.
- `items.csv`: item_id, category, description, price - 20 items, 7 with
  zero interactions (tests unseen-item handling).
- `users.csv`: user_id, age, age_band, region, preferred_category.

Regenerate with `python data/synthetic_demo/generate_demo_data.py`
(deterministic, no RNG - byte-identical output).

## Adapting to a real dataset later

Only `column_mapping` needs editing (or the CSV renamed to canonical
headers). Example for a dataset with columns `customer, product, qty, ts`:

```json
"interactions": {"user_id": "customer", "item_id": "product",
                 "interaction_value": "qty", "timestamp": "ts",
                 "required": ["customer", "product"]}
```
