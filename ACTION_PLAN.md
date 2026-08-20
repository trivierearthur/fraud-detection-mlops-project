# Action Plan

This is for you, not for whoever's reading the code with you. It explains what
changed in the repo, what's left to do before the defence, and what's worth
knowing for the oral even though nobody's grading it directly.

A few terms up front, since they'll come up a lot:

- **Lockfile** (`uv.lock`): a file that pins the *exact* version of every
  package you depend on, so "works on my machine" becomes "works on every
  machine." Without it, you and CI could silently end up with different
  package versions.
- **Linter**: a tool that reads your code without running it and flags likely
  mistakes (unused imports, comparing the wrong types, etc.). We use **ruff**.
- **Type checker**: a tool that checks your function signatures and usages are
  internally consistent (e.g. you're not passing a string where a number is
  expected) without running the code. We use **pyright**.
- **CI (Continuous Integration)**: a robot that runs your checks automatically
  every time you push code, so mistakes get caught before they reach `main`.
- **Fixture** (pytest term): a reusable piece of test setup — here, `client`
  spins up a fake version of the Flask app so tests can hit it without a real
  server running.

Before diving in: `REQUIREMENTS.md` has a table mapping every brief
requirement to where it's satisfied in the repo — worth a skim now, and worth
re-reading right before the defence.

## How to work in this project now

1. Install the recommended VS Code extensions: open the project, VS Code will
   prompt you ("This workspace has extension recommendations") — click
   **Install All**. If it doesn't prompt, open the Extensions panel and search
   `@recommended`.
2. Install `uv` and `just` (one-time, see step 1 below if you haven't already).
3. Run:
   ```powershell
   just setup
   ```
   This creates a `.venv` folder (an isolated Python environment just for this
   project) and installs every dependency at the exact version in `uv.lock`.
4. From now on, every task is a `just` command. Run `just` with no arguments
   any time to see the full list with descriptions. The ones you'll use daily:

   | Command | What it does |
   |---|---|
   | `just train` | Train a model, save a new version |
   | `just serve` | Start the API + Streamlit UI |
   | `just test` | Run the test suite |
   | `just check` | Everything CI runs — format check, lint, type check, tests |
   | `just fmt` | Auto-format your code |
   | `just lint` | Auto-fix lint issues |

5. **What changes the moment you save a `.py` file in VS Code**: it gets
   auto-formatted (spacing, quotes, line breaks — you never do this by hand
   again) and its imports get auto-sorted. Problems ruff or pyright can't
   auto-fix show up as a red/yellow squiggly line right in the editor, with a
   description on hover — you find out at write-time, not when CI fails ten
   minutes later.

## Part 1 — What changed in your repo, and why

- **`pyproject.toml` + `uv.lock` replace `requirements.txt`.** Same packages,
  same versions you already had — now with every transitive dependency
  pinned too, and a single command (`just setup`) to reproduce the exact
  environment anywhere, including CI.
- **ruff and pyright are now configured and the whole codebase is clean
  against them** (`just check` passes). Most of the work was adding type
  hints to function signatures — this doesn't change what any function does,
  it just writes down what was already true.
- **`Justfile`** — the command reference table above. This is new; it didn't
  exist before.
- **`.vscode/settings.json` + `extensions.json`** — format-on-save,
  inline diagnostics, pytest discovery in the editor.
- **`.editorconfig` / `.gitattributes`** — keeps line endings consistent
  between your Windows machine and Linux CI so you don't get diffs made of
  pure whitespace noise.
- **`ci.yml` and `mlops.yml`** now use `uv` and `just check` instead of raw
  `pip install` — same logic, reproducible environment.
- **`mlops.yml` gained two things the brief asked for that were missing**:
  a monthly scheduled trigger (previously it only retrained when triggered by
  a push or manually), and a step that actually starts the API after
  retraining and sends it a real request, to prove the new model is live and
  answering — not just that the file loaded.
- **5 stale compiled `.pyc` files were removed from git tracking** — they were
  already supposed to be ignored (`.gitignore` covers `__pycache__/`) but had
  been committed before that rule existed.
- **`.vscode/` is no longer gitignored.** It was blocking `settings.json` and
  `extensions.json` from ever being committed, which defeats their purpose.

## Part 2 — Steps you need to take

### Step 1 — Install `uv` and `just` `[production-readiness]` (5 min)

Why: these are the two tools everything else in this plan depends on.

```powershell
winget install --id Casey.Just
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

How you know it worked: open a **new** terminal and run `just --version` and
`uv --version` — both should print a version number.

### Step 2 — Rotate the leaked API key `[bug fix]` (5 min)

**This is time-sensitive — do it before anything else.** Commit `a6ae2af`
added a real `.env` file containing `API_KEY=29091988`. It was deleted in a
later commit, but deleting a file doesn't delete it from git's history —
anyone who clones this repo can still recover that value with
`git show a6ae2af:.env`. Treat `29091988` as compromised.

```powershell
python -c "import secrets; print(secrets.token_hex(16))"
```

Copy the output into your local `.env` file (create it from `.env.example`
if you haven't) and, if you set `API_KEY` as a GitHub Actions secret, update
that too (repo Settings → Secrets and variables → Actions).

How you know it worked: the old value `29091988` no longer authenticates
against your API; `just test` still passes with the new value in `.env`.

*(We did not rewrite git history to scrub the old value — that requires a
force-push and rewrites every commit hash after it, which is riskier than
it's worth for a school project. Rotating the key makes the old value
worthless, which is what actually matters.)*

### Step 3 — Fix `predict.py`'s broken model path `[bug fix]` (10 min)

Why: `python src/predict.py` currently always crashes with
`FileNotFoundError`. [src/predict.py:10](src/predict.py#L10) hardcodes
`models/fraud_model.joblib` — a file that's never created. Training only ever
produces `fraud_model_v1.joblib`, `_v2`, `_v3`, etc., recorded in
`models/current_model.txt`.

`src/api.py`'s `load_current_model()` (around
[src/api.py:38](src/api.py#L38)) already does this correctly — read
`current_model.txt`, resolve the path from there. Mirror that pattern in
`predict.py`'s `load_model()`.

How you know it worked: `uv run python src/predict.py` runs and prints
predictions instead of crashing.

### Step 4 — Make `api.py`'s import consistent `[improvement]` (10 min)

Why: [src/api.py:8](src/api.py#L8) does `from auth import require_api_key` —
a bare import that only resolves because of how the app happens to be
launched today. Every other module (`train_model.py`, `preprocess.py`,
`predict.py`) uses `from src.xxx import ...`. It works right now, but it's
one refactor away from breaking in a confusing way.

Change it to `from src.auth import require_api_key`, then update
`run_app.py`'s `subprocess.Popen([sys.executable, "src/api.py"])` to
`subprocess.Popen([sys.executable, "-m", "src.api"])` so the module-style
import still resolves when launched.

How you know it worked: `just serve` still starts both the API and the UI,
and `/health` still responds.

### Step 5 — Document where your data came from `[required by the brief]` (10 min)

Why: the brief requires an "open data source" (task step 2), and the schema
in `data/raw/fraud_data.csv` looks like it's derived from a known Kaggle
fraud-detection dataset, but nothing in the repo says which one or links to
it. You'll very likely get asked "where's this data from?" in the defence —
have the answer ready in the README, not just in your head.

Add a short "Data source" section to `README.md` with the Kaggle (or
wherever) link and, if you modified/subsampled it, a one-line note on what
you changed.

How you know it worked: someone reading the README can find and download the
original dataset themselves.

## Production-readiness notes

The assessment is "From Model to Production," so here's how this would hold
up if it were a real system — separate from anything the brief actually
requires. Some of these are genuinely strong points worth raising unprompted
in the defence; none of them affect your grade directly, so spend time here
in proportion to what's left before the deadline.

| Finding | Risk | Fix |
|---|---|---|
| Leaked API key in git history | **Critical** | Covered in Step 2 above |
| `/predict` errors are caught broadly and never logged ([src/api.py:159-160](src/api.py#L159-L160): `except Exception: return ..., 500`) | **Important** — if something breaks in production at 2am, there is no log line explaining what. Right now a real bug and an expected error look identical to whoever's on call. | Add `logging.exception("Prediction failed")` before returning the 500 |
| No record of which model version produced a given prediction, and no log of predictions at all | **Important**, especially in a government-agency context — if a citizen disputes a fraud flag, nobody can reconstruct why the model decided that. This is one of the strongest points you can raise unprompted in the defence: you *know* auditability matters here even though the brief never asks for it, which is worth more than having built it. | Log each prediction (timestamp, input hash or key fields, model version, output) to a file or simple log line — don't log full PII-bearing payloads unmodified |
| Feature engineering is applied identically at training and serving time — one `TransactionFeatureTransformer`, used by both `train_model.py` and `api.py`/`predict.py` via the saved pipeline | **This is a strength, not a gap.** The single most common real-world MLOps failure is "training/serving skew" — the transformation logic quietly drifting apart between the two paths. Yours can't drift apart because there's only one copy of it. Say this explicitly in the defence; it's a deliberate design choice paying off, not an accident. | No action needed |
| Data validation at the API boundary (missing fields, wrong types, bad datetime format all rejected with a clear 400) | Already solid — no gap | No action needed |
| `run_app.py` shuts down both the API and the UI cleanly on Ctrl+C | Already solid | No action needed |
| Class imbalance handled via `class_weight="balanced"` + `StratifiedKFold` + model selection by PR-AUC rather than accuracy | Already solid, and a good thing to explain *why* in the defence (accuracy is nearly meaningless when fraud is a small minority of transactions) | No action needed |
| Dataset is synthetic (Kaggle-style), so there's no real PII exposure today | Not a current risk | If this ever ran on real citizen data, you'd need field-level redaction before logging and an access-control review — worth a sentence in the defence as "what would need to change for production," not something to build now |

## Senior-engineer design notes

These are the kind of comments a senior engineer would leave reading this
code over your shoulder — not bugs, not brief requirements, just things that
work fine at school scale and would need rethinking at real scale. Ordered by
how much they'd actually matter here. Items marked "touches business logic"
are yours to decide on, not something to hand to a tool.

1. **CSV for `data/processed/fraud_data_cleaned.csv` loses information on
   every read.** This file gets written once by `preprocess.py` and read
   repeatedly by `train_model.py`, `simulate_months.py`, and `monitoring.py`.
   CSV has no schema — every read has to re-guess column types, and a column
   like `city_pop` can silently flip between `int` and `float` depending on
   whether any row had a missing value that read. **Parquet** is the normal
   fix: it stores the schema and dtypes alongside the data, so what you wrote
   is exactly what you read back, and it's faster for a file this size.
   `data/raw/` can stay CSV since that's the format it arrived in from the
   source. Cost: change one `.to_csv()`/`.read_csv()` pair per file to
   `.to_parquet()`/`.read_parquet()` (needs the `pyarrow` package) — under an
   hour, and it's I/O plumbing, not modeling, so this doesn't touch business
   logic.

2. **`data/raw/fraud_data.csv` gets overwritten during retraining.**
   `mlops.yml`'s drift-triggered retrain does
   `cp data/monthly/month_04.csv data/raw/fraud_data.csv` — so after a drift
   month, "raw" data is no longer the original dataset, it's last month's
   simulated batch. This doesn't affect what's committed to the repo (each CI
   run starts from a fresh checkout), but it means each retrain trains on
   *only* the newest month rather than accumulated history, and there's no
   record of exactly which data trained which model version. This is a
   legitimate design choice for a demo (it makes the drift-and-retrain cycle
   easy to show), but it's worth being able to explain *why* you chose
   "latest month only" over "accumulate history" if asked. **Touches business
   logic** — your call, not something to change without thinking through the
   retraining strategy you actually want.

3. **Model files are raw `joblib`/pickle with no version metadata attached.**
   Two risks: `joblib.load()` executes arbitrary code, so never load a
   `.joblib` file from a source you don't trust; and a pickled model is
   tightly coupled to the exact library versions that created it (already
   pinned for you now via `uv.lock`, so this is lower-risk than it was, but
   still worth knowing). A stronger setup saves a small sidecar file
   (`fraud_model_v6.metadata.json`) recording the training data snapshot,
   library versions, and metrics next to each model. Cost: maybe 30 minutes,
   touches `save_model()` in `train_model.py`, but only adds a file — doesn't
   change what gets trained or how, so this one's low-risk if you want it.

4. **Configuration is scattered across the codebase as bare constants** —
   `HIGH_CARDINALITY_THRESHOLD` in `preprocess.py`, `SIGNIFICANCE_LEVEL` in
   `monitoring.py`, `ROWS_PER_MONTH`/`RANDOM_STATE` in `simulate_months.py`,
   the Flask host/port in `api.py`. Fine at this size, but if the project grew
   you'd want one place to look. Note that several of these
   (`SIGNIFICANCE_LEVEL`, `RANDOM_STATE`) are modeling decisions, not just
   config — moving them is safe, but changing their *values* is a DS decision
   and **touches business logic**.

5. **The 9 EDA plot images in `notebooks/eda_plots/` are committed to git.**
   They're generated output, not source — regenerating them is one command
   (`uv run python notebooks/eda.py`). Committing generated files bloats the
   repo over time and means the images silently go stale if the analysis
   changes. Add `notebooks/eda_plots/` to `.gitignore` and regenerate on
   demand. Low cost, pure housekeeping.

6. **No retention policy for old model versions.** Every training run adds a
   new ~32 MB `.joblib` file and never removes the old ones — after a year of
   monthly retraining that's ~400 MB in the repo. Fine for a school project's
   timeframe; worth knowing this would need a pruning policy (keep last N
   versions, or move old ones to external storage) before it ran unattended
   for real. No action needed now.

## If something breaks

- **`just check` fails on the format step**: run `just fmt` to auto-fix it,
  then try again.
- **It fails on lint with a code like `ANN001` or `F401`**: the letters are
  the rule category (`ANN` = missing type annotation, `F` = actual likely bug
  like an unused import, `E`/`W` = style, `B` = common bug pattern). Run
  `just lint` to auto-fix what can be fixed; for the rest, the terminal output
  tells you the exact file, line, and what's wrong. You can look up any code
  at `https://docs.astral.sh/ruff/rules/` if the message alone isn't enough.
- **It fails on typecheck**: pyright prints `file.py:line:column - error:
  message`. Click the file:line in VS Code's terminal to jump straight there.
  If the message mentions a package you didn't write (pandas, sklearn) rather
  than your own code, it's likely a stub-typing limitation, not a real bug —
  ask before spending too long on it.
- **It fails on test with `assert 500 == 200` or similar**: almost always
  means `API_KEY` isn't set. Make sure `.env` exists (copy `.env.example`)
  and has a value in it.
- **Nothing seems to work and you're not sure why**: delete `.venv` and run
  `just setup` again — this rebuilds your environment from scratch and fixes
  most "it worked yesterday" problems.
