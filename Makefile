.PHONY: help docker-up docker-down ingest eval report test test-all lint format clean

PYTHON := python
PYTEST  := pytest

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

docker-up:  ## Start OpenSearch via Docker Compose
	docker compose -f docker/docker-compose.yml up -d
	@echo "Waiting for OpenSearch to be ready..."
	@until curl -sf http://localhost:9200/_cluster/health > /dev/null 2>&1; do sleep 2; done
	@echo "✅ OpenSearch is ready at http://localhost:9200"

docker-down:  ## Stop and remove containers
	docker compose -f docker/docker-compose.yml down -v

ingest:  ## Load knowledge base documents into the vector store
	$(PYTHON) -m rag_eval.ingest.pipeline

eval:  ## Run the full evaluation pipeline
	$(PYTHON) -m rag_eval.pipeline.evaluator

report:  ## Print the most recent eval report
	@latest=$$(ls -t reports/*.json 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then echo "No reports found. Run 'make eval' first."; \
	else $(PYTHON) -m rag_eval.reporting.json_reporter --file $$latest; fi

test:  ## Run unit tests (no external services required)
	$(PYTEST) tests/unit/ -v -m "not integration and not opensearch"

test-all:  ## Run all tests including integration
	$(PYTEST) tests/ -v

test-integration:  ## Run integration tests (requires Docker)
	$(PYTEST) tests/integration/ -v

lint:  ## Lint with ruff and check formatting with black
	ruff check rag_eval/ tests/
	black --check rag_eval/ tests/

format:  ## Auto-format code with black
	black rag_eval/ tests/

clean:  ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf reports/ chroma_db/
