.PHONY: sync check clean

sync:
	./scripts/sync-runtime.sh

check:
	./scripts/check-runtime.sh

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
