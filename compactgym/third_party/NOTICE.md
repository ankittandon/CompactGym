# Third-party code

conpact and cliff are MIT licensed (codex is Apache-2.0); license texts sit next to the vendored files.

| Dir | Source | Files |
|---|---|---|
| `conpact/` | github.com/ZhiqiEliWang/compaction-integrity @ e38e298 (c) 2026 Zhiqi Wang | `constraints.py`, `prompts.py` (verbatim); `extractor.py` (extracted from `compaction_integrity/prompts.py`) |
| `cliff/` | github.com/nguyenvuthientrang/cliffcompaction @ b48d660 (c) 2026 Trang Nguyen | `cliff.py`, `config.py`, `hashing.py`, `dialect_base.py`, `dialect_openai_chat.py` (relative imports rewritten only) |

`compactors/cliff.py` and `compactors/constraint_extractor.py` adapt these.
| `codex/` | github.com/openai/codex (Apache-2.0), `codex-rs/prompts/templates/compact/prompt.md` | `compact_prompt.md` (verbatim); used by `compactors/threshold.py::ThresholdCodex` |
