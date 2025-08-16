PY=python

help:
	@echo 'Targets:'
	@echo '  make etl       - run ETL on sample/raw data'
	@echo '  make features  - build engineered feature table'
	@echo '  make train     - train RF/GBT and save best model'
	@echo '  make cluster   - run KMeans sweep'
	@echo '  make notebook  - start Jupyter'

etl:
	$(PY) -m src.etl

features:
	$(PY) -m src.features

train:
	$(PY) -m src.train

cluster:
	$(PY) -m src.cluster

notebook:
	jupyter notebook
