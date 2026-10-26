# KNOWN FACTS - Real LMArena Battle Dataset (`arena_battles_real.csv`)

Generated: 2026-10-19 09:56 UTC by `scripts/build_arena_dataset.py`.

## Provenance

Real, publicly released human-preference votes collected by LMArena
(Chatbot Arena). No synthetic rows are present in this file.

| source | license | rows scanned | kept |
|---|---|---|---|
| `lmarena-ai/arena-human-preference-100k` | see dataset card | 106134 | 3255 |
| `lmarena-ai/arena-human-preference-140k` | cc-by-4.0 | 135634 | 700 |

- URL: https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k
- URL: https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k

## Filters applied (in order)

1. Language must be English (`language == 'English'` / `'en'`).
2. `winner` must be `model_a` or `model_b` (ties and `both_bad` dropped).
3. Both battle sides must alias-map onto the engine's 7 candidate models.
4. First user prompt must be >= 10 characters.
5. Exact duplicates of (prompt, model_a, model_b, winner) removed.

## Candidate alias mapping (raw Arena name -> canonical id)

- `gpt-4o` <- `gpt-4o`, `gpt-4o-2024-05-13`, `gpt-4o-2024-08-06`, `gpt-4o-2024-11-20`, `chatgpt-4o-latest`, `chatgpt-4o-latest-20250326`
- `claude-3-5-sonnet` <- `claude-3-5-sonnet`, `claude-3-5-sonnet-20240620`, `claude-3-5-sonnet-20241022`
- `deepseek-r1` <- `deepseek-r1`, `deepseek-r1-0528`
- `qwen3-235b-a22b` <- `qwen3-235b-a22b`, `qwen3-235b-a22b-no-thinking`, `qwen3-235b-a22b-thinking-2507`, `qwen3-235b-a22b-instruct-2507`
- `gemini-1.5-pro` <- `gemini-1.5-pro`, `gemini-1.5-pro-api-0514`, `gemini-1.5-pro-exp-0801`
- `llama-3.1-70b` <- `llama-3.1-70b`, `llama-3.1-70b-instruct`
- `gpt-4o-mini` <- `gpt-4o-mini`, `gpt-4o-mini-2024-07-18`

### Honest notes

- `qwen-2.5-coder-32b` (the project's original candidate) has **zero** battles in
  either source, so the lineup was swapped to `qwen3-235b-a22b` before this file
  was built. See `docs/multi_criteria_llm_routing_architecture.md`.
- Version snapshots of the same model family (e.g. `gpt-4o-2024-05-13` and
  `chatgpt-4o-latest`) are merged into one canonical candidate; per-snapshot
  differences are therefore not modeled.
- `gemini-1.5-pro` and `llama-3.1-70b` battles come only from the 2024 source;
  `deepseek-r1` and `qwen3-235b-a22b` only from the 2025 source. The two eras are
  linked through `gpt-4o`, `gpt-4o-mini` and `claude-3-5-sonnet`, which appear in both.

## Row counts

- **total battles: 3945**
- exact duplicates removed: 10
- per source: {'arena-100k': 3245, 'arena-140k': 700}
- battles per model (appearances): {'claude-3-5-sonnet': 2016, 'llama-3.1-70b': 611, 'gemini-1.5-pro': 1709, 'gpt-4o': 2300, 'gpt-4o-mini': 602, 'qwen3-235b-a22b': 417, 'deepseek-r1': 235}

## Head-to-head pair coverage

- claude-3-5-sonnet vs gpt-4o: 895
- claude-3-5-sonnet vs gemini-1.5-pro: 723
- gemini-1.5-pro vs gpt-4o: 664
- gpt-4o vs llama-3.1-70b: 312
- gpt-4o vs gpt-4o-mini: 236
- gemini-1.5-pro vs gpt-4o-mini: 175
- gemini-1.5-pro vs llama-3.1-70b: 147
- claude-3-5-sonnet vs qwen3-235b-a22b: 145
- gpt-4o vs qwen3-235b-a22b: 137
- deepseek-r1 vs qwen3-235b-a22b: 135
- claude-3-5-sonnet vs gpt-4o-mini: 124
- claude-3-5-sonnet vs llama-3.1-70b: 85
- gpt-4o-mini vs llama-3.1-70b: 67
- deepseek-r1 vs gpt-4o: 56
- claude-3-5-sonnet vs deepseek-r1: 44
