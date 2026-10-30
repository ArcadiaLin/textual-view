PYTHON := .venv/bin/python
DATA := ../../references/datasets/stark
STARK_REPO := ../../references/repos/stark
INDEX := ../../data/stark-prime/bm25
RUN := ../../runs/stark-prime/$(shell date -u +%Y%m%dT%H%M%SZ)

.PHONY: setup index test pilot
setup:
	uv sync --frozen --python /usr/bin/python3

index:
	$(PYTHON) retrieval.py --archive $(DATA)/skb/prime/processed.zip --index $(INDEX) --stark-repo $(STARK_REPO)

test:
	$(PYTHON) -m unittest discover -s tests -v

pilot:
	$(PYTHON) batch.py --data-root $(DATA) --index $(INDEX) --output $(RUN) --count 10 --workers 2 --timeout 180 --max-turns 20
