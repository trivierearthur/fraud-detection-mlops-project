set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# List available recipes
default:
    @just --list

# Install Python, create the virtual environment, and install all dependencies
setup:
    uv sync

# Format all code (rewrites files)
fmt:
    uv run ruff format .

# Check formatting without changing files (used by `check` / CI)
fmt-check:
    uv run ruff format --check .

# Lint and auto-fix what can be fixed safely
lint:
    uv run ruff check . --fix

# Type-check the codebase with pyright
typecheck:
    uv run pyright

# Run the test suite
test:
    uv run pytest

# Run everything CI runs: formatting, linting, type checking, tests
check:
    uv run ruff format --check .
    uv run ruff check .
    uv run pyright
    uv run pytest

# Train a new model version (saves to models/, updates current_model.txt)
train:
    uv run python main.py

# Serve the API (Flask) and the UI (Streamlit)
serve:
    uv run python run_app.py

# Generate 12 months of simulated data with injected drift
simulate:
    uv run python src/simulate_months.py

# Check simulated monthly data for drift against the training reference
monitor:
    uv run python src/monitoring.py

# Check a single simulated month for drift (used by the mlops.yml workflow)
monitor-month month:
    uv run python src/monitoring.py --month {{ month }}

# Retrain the model and save the next version
retrain:
    uv run python -m src.retrain

# Remove cache directories (__pycache__, .pytest_cache, .ruff_cache)
clean:
    uv run python scripts/clean.py
