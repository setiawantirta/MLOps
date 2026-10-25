.PHONY: data latih uji api drift promosi bersih semua

data:
	python buat_dataset.py --keluaran data/raw --seed 42

latih:
	python src/train.py --config conf/params.yaml --keluaran artifacts

uji:
	python -m pytest tests -q

api:
	uvicorn src.api:app --reload --port 8000

drift:
	python src/monitor.py --config conf/params.yaml --artefak artifacts

promosi:
	python src/promote.py --juara artifacts_juara --penantang artifacts_penantang

kartu:
	python src/model_card.py --artefak artifacts --keluaran laporan/model_card.md

semua: data latih uji drift

bersih:
	rm -rf artifacts artifacts_* laporan data/processed .pytest_cache
