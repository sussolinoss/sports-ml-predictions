# Code Style Guide (Python)

ML/data project. Same principles I use in Java/Kotlin, adapted to Python:
readable, predictable, fail-fast, no overengineering.

---

## 1. Naming
- Descriptive names. `model`, `calibrator`, `features` — not `m`, `cal`, `f`.
- Short loop/idiom names are fine where standard (`i`, `df`, `ax`).
- No type prefixes/suffixes (`*Impl`, `IFoo`). The name says what it is.
- Functions are verbs (`build_features`, `load_results`), values are nouns.

## 2. Comments
- Minimal, in English. Explain the **why**, never the **what**.
- No banner lines (`# ======`), no scratch notes (`exp 36 BOCCIATO`), no emoji,
  no dead `TODO`. If a line needs a comment to be understood, prefer clearer code.
- One short module docstring at the top: what it does and how to run it.

## 3. Validation (fail-fast)
- Validate at the boundary (entry points, public functions, CLI args), not in
  every inner call.
- Raise early with a clear message; do not silently continue on bad input.

```python
if not path.exists():
    raise FileNotFoundError(f"dataset not found: {path}")
```

## 4. Functions and structure
- One responsibility per function. Split when it does two things.
- Return values, avoid hidden mutation of arguments.
- No global mutable state. Pass dependencies in; if a module-level constant is
  needed, keep it immutable (tuple/frozenset, UPPER_CASE).

## 5. Data structures
- Use the right one: `dict` for keyed access, `list` for sequences,
  `set` for membership, `deque(maxlen=...)` for rolling windows.
- Do not expose internal mutable state through a getter without copying when it
  matters.

## 6. Anti-leakage (this project specifically)
- Read the state, write the feature row, **then** update the state. Never the
  other way around. Getting this order wrong leaks the result into the features.
- Splits are by time, never random.

## 7. Exceptions
- No bare `except:` and no `except Exception: pass`. Catch the specific error.
- Either handle it or re-raise with context. At a top-level loop, log and
  continue is fine — say why in a comment.

```python
try:
    data = json.loads(path.read_text())
except json.JSONDecodeError:
    continue  # skip malformed race file, keep processing the rest
```

## 8. File I/O for artifacts
- For models and result files that matter, write to a temp path then rename,
  so a crash mid-write does not leave a corrupt file.
- Stream large files; do not read huge dumps fully into memory when avoidable.

## 9. Performance
- Do not optimise without measuring. Prefer simple code.
- Vectorise with numpy/pandas instead of Python loops in hot paths.
- No premature micro-optimisation.

## 10. Reproducibility
- Fixed random seeds where results depend on them; state the seed.
- The published number is the one the code reproduces on a plain run, not a
  best-of selection. If a best-of-seed number is reported, label it as such.

---

Anti-patterns to avoid: cryptic names, banner/scratch comments, validation
duplicated everywhere, bare excepts, global mutable singletons, random splits on
time-series data, claiming a number the code does not reproduce.
