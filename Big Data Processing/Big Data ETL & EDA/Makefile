
run-all:
	PYTHONPATH=src python -m portfolio_etl.cli ingest --base-path data/raw/samples
	PYTHONPATH=src python -m portfolio_etl.cli transform --base-path data/raw/samples
	PYTHONPATH=src python -m portfolio_etl.cli queries --base-path data/raw/samples --out-dir data/processed --fig-dir reports/figures

download-data:
	mkdir -p data/raw
	# Replace this with a download link or copy the full transactions.csv here
	echo "Place transactions.csv in data/raw/"
