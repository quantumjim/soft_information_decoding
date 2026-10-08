RAW_TAR ?= softinfo-raw-v1.tar

.PHONY: setup figures data hardware sim rerun-all test clean

setup:        ## install the environment
	uv sync

figures:      ## all paper figures and Table I -> figures/
	uv run python scripts/figures.py

data:         ## unpack the raw IQ archive (8.5 GB) -> data/raw: make data RAW_TAR=path/to/softinfo-raw-v1.tar
	tar -xf $(RAW_TAR) -C data

hardware:     ## decode one real IBM job, d = 3..11 (needs data)
	uv run python scripts/rerun.py hardware --states Z0 --rounds 50 --max-jobs 1 --distances 3 5 7 9 11

sim:          ## small simulation of the device (needs data)
	uv run python scripts/rerun.py sim --state X0 --shots 1000 --distances 3 5 7 9 11

rerun-all:    ## full re-decoding of all hardware jobs (about a day on 24 cores)
	uv run python scripts/rerun.py hardware --workers $(shell nproc)

test:         ## end-to-end tests (data-dependent ones are skipped without data/raw)
	uv run pytest -q

clean:
	rm -rf figures data/rerun
