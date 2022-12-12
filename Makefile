install: ## [Local development] Upgrade pip, install requirements, install package.
	python -m pip install -U pip
	python -m pip install -e .

black: ## [Local development] Auto-format python code using black
	python -m black -l 120 .

lint: ## [Local development] Run mypy, pylint and black
	python -m black --check -l 120 src
	python -m mypy src
	python -m pylint src

install-training:
	python -m pip install -r requirements-training.txt

install-test: ## [Local development] Install test requirements
	python -m pip install -r requirements-test.txt

test: ## [Local development] Run unit tests
	python -m pytest -x -s -v tests
