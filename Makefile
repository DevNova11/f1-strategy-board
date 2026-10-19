# F1 Strategy Board - common tasks. On Windows without make, use run.ps1.
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.PHONY: setup data sample build report notebook test run clean

setup:            ## create the virtual environment and install pinned dependencies
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

data:             ## download real data (Jolpica + FastF1) into data/raw
	$(PY) -m src.fetch

sample:           ## rebuild the small bundled sample from data/raw
	$(PY) -m src.fetch --make-sample

build:            ## run sources -> ingest -> clean and cache the analysis tables
	$(PY) -m src.pipeline

report:           ## generate reports/eda_report.html
	$(PY) -m src.eda.report

notebook:         ## rebuild and execute notebooks/01_eda_walkthrough.ipynb
	$(PY) notebooks/build_notebook.py

test:             ## run the test suite
	$(PY) -m pytest

run:              ## launch the dashboard
	$(PY) -m streamlit run app/Home.py

clean:            ## remove derived data (keeps data/raw and data/sample)
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('data/processed', 'data/synthetic')]"
