# vector_search_101 — what each target is for: .claude/CLAUDE.md §4.
.DEFAULT_GOAL := help
.PHONY: help setup sync data lint format check typecheck test test-cov test-offline \
        notebooks nb run lab clean reset ci dev

# `make setup` may have just installed uv into ~/.local/bin, which is not on PATH yet.
UV := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)

KERNEL := vector-search-101

# nbconvert's timeout applies per CELL. The PRD budget (< 7 min per notebook) is
# enforced separately by timing each whole notebook against NB_BUDGET (seconds).
CELL_TIMEOUT := 420
NB_BUDGET := 420
NB_EXEC = $(UV) run jupyter nbconvert --to notebook --execute --inplace \
	--ExecutePreprocessor.timeout=$(CELL_TIMEOUT)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup: uv, deps, Jupyter kernel
	@command -v uv >/dev/null 2>&1 || [ -x "$(HOME)/.local/bin/uv" ] || \
		curl -LsSf https://astral.sh/uv/install.sh | sh
	$(UV) sync --all-extras
	$(UV) run python -m ipykernel install --user --name $(KERNEL) --display-name "Python ($(KERNEL))"
	@printf '\n✅ Ready. Run "make data" then "make test".\n'

sync: ## Sync deps (all extras)
	$(UV) sync --all-extras

data: ## Build/cache embedding + synthetic vectors and exact ground truth
	$(UV) run python -m src.datasets.vectors
	@printf '✅ Vectors cached in data/vectors/\n'

lint: format check typecheck ## All linters (format + check + typecheck)

format: ## ruff format
	$(UV) run ruff format src/ tests/ app/

check: ## ruff check --fix
	$(UV) run ruff check --fix src/ tests/ app/

typecheck: ## ty check src/
	$(UV) run ty check src/

test: ## pytest
	$(UV) run pytest

test-cov: ## pytest with coverage
	$(UV) run pytest --cov=src --cov-report=term-missing

test-offline: ## pytest with no Hub access (deselects @pytest.mark.network)
	HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(UV) run pytest -m "not network"

notebooks: data ## Execute all notebooks in place, timed; fails if any exceeds 7 min
	@set -e; found=0; over=""; \
	for nb in notebooks/0*.ipynb; do \
		[ -e "$$nb" ] || continue; \
		case "$$nb" in *.nbconvert.ipynb) continue ;; esac; \
		found=1; start=$$(date +%s); \
		echo "▶ Executing $$nb ..."; \
		$(NB_EXEC) "$$nb"; \
		secs=$$(( $$(date +%s) - start )); \
		echo "  ⏱  $$nb: $${secs}s"; \
		if [ $$secs -gt $(NB_BUDGET) ]; then over="$$over $$nb ($${secs}s)"; fi; \
	done; \
	if [ $$found -eq 0 ]; then echo "No notebooks in notebooks/ yet."; exit 0; fi; \
	if [ -n "$$over" ]; then echo "⚠️  Over the $(NB_BUDGET)s per-notebook budget:$$over"; exit 1; fi; \
	printf '\n✅ All notebooks executed within budget.\n'

nb: ## Execute one notebook in place, timed: make nb NB=04
	@test -n "$(NB)" || { echo "usage: make nb NB=04"; exit 2; }; \
	set -- notebooks/$(NB)_*.ipynb; \
	[ -e "$$1" ] || { echo "No notebook matches notebooks/$(NB)_*.ipynb"; exit 2; }; \
	start=$$(date +%s); echo "▶ Executing $$1 ..."; \
	$(NB_EXEC) "$$1" || exit 1; \
	secs=$$(( $$(date +%s) - start )); \
	echo "  ⏱  $$1: $${secs}s (budget $(NB_BUDGET)s)"; \
	[ $$secs -le $(NB_BUDGET) ] || { echo "⚠️  Over budget"; exit 1; }

run: ## Streamlit app on :8501
	$(UV) run streamlit run app/streamlit_app.py --server.port 8501

lab: ## JupyterLab on :8888
	$(UV) run jupyter lab --no-browser --port 8888

clean: ## Remove caches, coverage artefacts, stray nbconvert copies
	find . -path ./.venv -prune -o -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov .coverage .coverage.* .pytest_cache .ruff_cache .ty_cache
	find notebooks -name '*.nbconvert.ipynb' -delete 2>/dev/null || true
	@printf '🧹 Cleaned.\n'

reset: clean ## Full reset: also drop .venv and cached vectors/indexes
	rm -rf .venv data/vectors data/indexes

ci: sync lint test ## What CI runs: sync + lint + test

dev: lint test ## Fast loop: lint + test
