# This makefile has been created to help developers perform common actions.
# Most actions assume it is operating in a virtual environment where the
# python command links to the appropriate virtual environment Python.

# Do not remove this block. It is used by the 'help' rule when
# constructing the help output.
# help:
# help: stylegan2 Makefile help
# help:

# help: help                           - display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: init                          - create an environment for development
.PHONY: init
init:
	@poetry run pip install -U pip
	@poetry install -E cpu

# help: init.gpu                      - create an environment for development
.PHONY: init.gpu
init.gpu:
	@poetry run pip install -U pip
	@poetry install
	@poetry run pip install jaxlib==0.1.57+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# help: clean                          - clean all files using .gitignore rules
.PHONY: clean
clean:
	@git clean -X -f -d

# help: scrub                          - clean all files, even untracked files
.PHONY: scrub
scrub:
	@git clean -x -f -d


# help: test                           - run tests
.PHONY: test
test:
	@poetry run pytest stylegan2 tests


# help: style                          - perform code formatting
.PHONY: style
style:
	@poetry run isort stylegan2 tests
	@poetry run black --include .py --exclude ".pyc|.pyi|.so" stylegan2 tests
	@poetry run black --pyi --include .pyi --exclude ".pyc|.py|.so" stylegan2 tests


# help: check                          - perform linting checks
.PHONY: check
check:
	@poetry run isort --check stylegan2 tests
	@poetry run black --check --include .py --exclude ".pyc|.pyi|.so" stylegan2 tests
	@poetry run black --check --pyi --include .pyi --exclude ".pyc|.py|.so" stylegan2 tests
	@poetry run pylint stylegan2
	@poetry run mypy -p stylegan2


# Keep these lines at the end of the file to retain nice help
# output formatting.
# help:
