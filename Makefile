.PHONY: dev test eval seed clean venv frontend-deps

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	@test -d $(VENV) || python3.11 -m venv $(VENV)
	@$(PIP) install --quiet -r backend/requirements.txt

frontend-deps:
	@test -d frontend/node_modules || (cd frontend && npm install)

# One command: backend on :8000, frontend on :3000
dev: venv frontend-deps
	./dev.sh

# Everything: engine golden tests, DSL validation, API tests, offline LLM tests,
# then the frontend typecheck.
test: venv
	$(PY) -m pytest backend/tests -q
	cd frontend && npm run typecheck

# Live LLM eval harness for the CONFIGURED model — needs that provider's key
eval: venv
	HONEST_RUN_LLM_EVALS=1 $(PY) -m pytest backend/tests/test_llm_evals.py -q -rs

# Compare models across providers on accuracy AND measured cost.
#   make bench
#   make bench MODELS=gpt-5-nano,claude-sonnet-5
#   make bench CASES=8
bench: venv
	$(PY) scripts/bench_models.py \
	  $(if $(MODELS),--models $(MODELS),) $(if $(CASES),--cases $(CASES),)

# Import the frozen fixture CSVs into the parquet price cache and build the
# symbol master, so first run works offline.
seed: venv
	$(PY) scripts/seed_cache.py

clean:
	rm -rf backend/**/__pycache__ .pytest_cache frontend/.next
