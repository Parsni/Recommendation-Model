# Synthetic demo dataset — known facts (ground truth for tests)

Deterministic output of `generate_demo_data.py` (seeded by construction, no RNG).
**This data is placeholder data for functionality testing only. Performance
measured on it is NOT real-world effectiveness.**

## Structure
- 30 users `U001..U030`; 20 catalog items `I001..I020`.
- Three taste groups of 10 users each (`user_index % 3`): group A = U004k+1-style
  cycling `A,B,C,A,B,C...` starting U001=A.
  - Group A consumes exactly `I001,I002,I003,I004`
  - Group B consumes exactly `I005,I006,I007,I008`
  - Group C consumes exactly `I009,I010,I011,I012`
- Shared item `I013` is consumed by all 30 users.
- Items `I014..I020` have metadata but **zero interactions** (unseen catalog items).

## Exact counts (after cleaning)
- 150 unique user-item pairs; interaction table has 151 rows because one
  deliberately corrupted extra row (U007→I013 with non-numeric value, coerced
  to missing, matrix-aggregated to mean) exists.
- Popularity (distinct users per item): I013 = 30; I001..I012 = 10 each;
  I014..I020 = 0.
- Popularity top-5 for a user who has seen nothing: I013, I001, I002, I003, I004
  (ties broken by item_id ascending).
- Ratings: `3 + ((user_number + item_number) mod 3)` ∈ {3,4,5}.
  Examples: U001×I001 = 5, U001×I013 = 5, U002×I005 = 4, U002×I006 = 5.
- Timestamps: 2025-01-01 + ((user_index-1)*5 + k) days, k = 0..4.
  Per-user order: I013 first, then group items in reverse order, so each user's
  **last** interaction is their group's first item.
  Examples: U001 last = I001 (2025-01-05); U002 last = I005 (2025-01-10).

## Seeded data-quality defects (in interactions.csv, expected by validation tests)
1. Row index 152 (0-based): blank user_id → must be dropped as malformed.
2. Two exact duplicate rows (copies of original rows 3 and 40) → detected + deduplicated.
3. One row U007→I013 with interaction_value "great" → non-numeric, coerced to NaN,
   row reported in validation.

## Association ground truth
- Every group-A user consumed I001 and I002 together → rule I001→I002 has
  support = 10/30 ≈ 0.333, confidence = 1.0 (within training windows that
  contain both). I005→I001 has confidence 0 (different groups).
