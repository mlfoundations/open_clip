install: ## [Local development] Upgrade pip, install requirements, install package.
	python3 -m pip install -U pip
	python3 -m pip install -e .

install-training:
	python3 -m pip install -r requirements-training.txt

install-test: ## [Local development] Install test requirements
	python3 -m pip install -r requirements-test.txt

test: ## [Local development] Run unit tests
	python3 -m pytest -x -s -v tests
