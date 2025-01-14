# ==================================================
# configuration & variables
# ==================================================

# !!! MODIFY AT LEAST THIS PART TO SUIT YOUR PROJECT !!!
# it assumes that the source is in a directory named the same as the package name
# this also gets passed to some other places
PACKAGE_NAME := decoding-gpt

# for checking you are on the right branch when publishing
PUBLISH_BRANCH := main

# where to put docs
DOCS_DIR := docs

# where to put the coverage reports
# note that this will be published with the docs!
# modify the `docs` targets and `.gitignore` if you don't want that
COVERAGE_REPORTS_DIR := docs/coverage

# where the tests are, for pytest
TESTS_DIR := tests/

# tests temp directory to clean up. will remove this in `make clean`
TESTS_TEMP_DIR := _temp/

# probably don't change these:
# --------------------------------------------------

# where the pyproject.toml file is. no idea why you would change this but just in case
PYPROJECT := pyproject.toml

# requirements.txt files for base package, all extras, dev, and all
REQ_LOCATION := .github/requirements

# local files (don't push this to git)
LOCAL_DIR := .github/local

# will print this token when publishing. make sure not to commit this file!!!
PYPI_TOKEN_FILE := $(LOCAL_DIR)/.pypi-token

# version files
VERSIONS_DIR := .github/versions

# the last version that was auto-uploaded. will use this to create a commit log for version tag
# see `gen-commit-log` target
LAST_VERSION_FILE := $(VERSIONS_DIR)/.lastversion

# current version (writing to file needed due to shell escaping issues)
VERSION_FILE := $(VERSIONS_DIR)/.version

# base python to use. Will add `uv run` in front of this if `RUN_GLOBAL` is not set to 1
PYTHON_BASE := python

# where the commit log will be stored
COMMIT_LOG_FILE := $(LOCAL_DIR)/.commit_log

# pandoc commands (for docs)
PANDOC ?= pandoc

# version vars - extracted automatically from `pyproject.toml`, `$(LAST_VERSION_FILE)`, and $(PYTHON)
# --------------------------------------------------

# assuming your `pyproject.toml` has a line that looks like `version = "0.0.1"`, `gen-version-info` will extract this
VERSION := NULL
# `gen-version-info` will read the last version from `$(LAST_VERSION_FILE)`, or `NULL` if it doesn't exist
LAST_VERSION := NULL
# get the python version, now that we have picked the python command
PYTHON_VERSION := NULL

# cuda version
# --------------------------------------------------
# 0 or 1
CUDA_PRESENT :=
# a version like "12.4" or "NULL"
CUDA_VERSION := NULL
# a version like "124" or "NULL"
CUDA_VERSION_SHORT := NULL


# python scripts we want to use inside the makefile
# --------------------------------------------------

# create commands for exporting requirements as specified in `pyproject.toml:tool.uv-exports.exports`
define EXPORT_SCRIPT
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from pathlib import Path
from typing import Union, List, Optional

pyproject_path: Path = Path(sys.argv[1])
output_dir: Path = Path(sys.argv[2])

with open(pyproject_path, 'rb') as f:
	pyproject_data: dict = tomllib.load(f)

# all available groups
all_groups: List[str] = list(pyproject_data.get('dependency-groups', {}).keys())
all_extras: List[str] = list(pyproject_data.get('project', {}).get('optional-dependencies', {}).keys())

# options for exporting
export_opts: dict = pyproject_data.get('tool', {}).get('uv-exports', {})

# what are we exporting?
exports: List[str] = export_opts.get('exports', [])
if not exports:
	exports = [{'name': 'all', 'groups': [], 'extras': [], 'options': []}]

# export each configuration
for export in exports:
	# get name and validate
	name = export.get('name')
	if not name or not name.isalnum():
		print(f"Export configuration missing valid 'name' field {export}", file=sys.stderr)
		continue

	# get other options with default fallbacks
	filename: str = export.get('filename') or f"requirements-{name}.txt"
	groups: Union[List[str], bool, None] = export.get('groups', None)
	extras: Union[List[str], bool] = export.get('extras', [])
	options: List[str] = export.get('options', [])

	# init command
	cmd: List[str] = ['uv', 'export'] + export_opts.get('args', [])

	# handle groups
	if groups is not None:
		groups_list: List[str] = []
		if isinstance(groups, bool):
			if groups:
				groups_list = all_groups.copy()
		else:
			groups_list = groups
		
		for group in all_groups:
			if group in groups_list:
				cmd.extend(['--group', group])
			else:
				cmd.extend(['--no-group', group])

	# handle extras
	extras_list: List[str] = []
	if isinstance(extras, bool):
		if extras:
			extras_list = all_extras.copy()
	else:
		extras_list = extras

	for extra in extras_list:
		cmd.extend(['--extra', extra])

	cmd.extend(options)

	output_path = output_dir / filename
	print(f"{' '.join(cmd)} > {output_path.as_posix()}")
endef

export EXPORT_SCRIPT

# get the version from `pyproject.toml:project.version`
define GET_VERSION_SCRIPT
import sys

try:
	if sys.version_info >= (3, 11):
		import tomllib
	else:
		import tomli as tomllib

	pyproject_path = '$(PYPROJECT)'

	with open(pyproject_path, 'rb') as f:
		pyproject_data = tomllib.load(f)

	print('v' + pyproject_data['project']['version'], end='')
except Exception as e:
	print('NULL', end='')
	sys.exit(1)
endef

export GET_VERSION_SCRIPT


# get the commit log since the last version from `$(LAST_VERSION_FILE)`
define GET_COMMIT_LOG_SCRIPT
import subprocess
import sys

last_version = sys.argv[1].strip()
commit_log_file = '$(COMMIT_LOG_FILE)'

if last_version == 'NULL':
    print('!!! ERROR !!!', file=sys.stderr)
    print('LAST_VERSION is NULL, can\'t get commit log!', file=sys.stderr)
    sys.exit(1)

try:
    log_cmd = ['git', 'log', f'{last_version}..HEAD', '--pretty=format:- %s (%h)']
    commits = subprocess.check_output(log_cmd).decode('utf-8').strip().split('\n')
    with open(commit_log_file, 'w') as f:
        f.write('\n'.join(reversed(commits)))
except subprocess.CalledProcessError as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
endef

export GET_COMMIT_LOG_SCRIPT

# get cuda information and whether torch sees it
define CHECK_TORCH_SCRIPT
import os
import sys
print(f'python version: {sys.version}')
print(f"\tpython executable path: {str(sys.executable)}")
print(f"\tsys_platform: {sys.platform}")
print(f'\tcurrent working directory: {os.getcwd()}')
print(f'\tHost name: {os.name}')
print(f'\tCPU count: {os.cpu_count()}')
print()

try:
	import torch
except Exception as e:
	print('ERROR: error importing torch, terminating        ')
	print('-'*50)
	raise e
	sys.exit(1)

print(f'torch version: {torch.__version__}')

print(f'\t{torch.cuda.is_available() = }')

if torch.cuda.is_available():
	# print('\tCUDA is available on torch')
	print(f'\tCUDA version via torch: {torch.version.cuda}')

	if torch.cuda.device_count() > 0:
		print(f"\tcurrent device: {torch.cuda.current_device() = }\n")
		n_devices: int = torch.cuda.device_count()
		print(f"detected {n_devices = }")
		for current_device in range(n_devices):
			try:
				# print(f'checking current device {current_device} of {torch.cuda.device_count()} devices')
				print(f'\tdevice {current_device}')
				dev_prop = torch.cuda.get_device_properties(torch.device(0))
				print(f'\t    name:                   {dev_prop.name}')
				print(f'\t    version:                {dev_prop.major}.{dev_prop.minor}')
				print(f'\t    total_memory:           {dev_prop.total_memory} ({dev_prop.total_memory:.1e})')
				print(f'\t    multi_processor_count:  {dev_prop.multi_processor_count}')
				print(f'\t    is_integrated:          {dev_prop.is_integrated}')
				print(f'\t    is_multi_gpu_board:     {dev_prop.is_multi_gpu_board}')
				print(f'\t')
			except Exception as e:
				print(f'Exception when trying to get properties of device {current_device}')
				raise e
		sys.exit(0)
	else:
		print(f'ERROR: {torch.cuda.device_count()} devices detected, invalid')
		print('-'*50)
		sys.exit(1)

else:
	print('ERROR: CUDA is NOT available, terminating')
	print('-'*50)
	sys.exit(1)
endef

export CHECK_TORCH_SCRIPT


# ==================================================
# reading command line options
# ==================================================

# for formatting or something, we might want to run python without uv
# RUN_GLOBAL=1 to use global `PYTHON_BASE` instead of `uv run $(PYTHON_BASE)`
RUN_GLOBAL ?= 0

ifeq ($(RUN_GLOBAL),0)
	PYTHON = uv run $(PYTHON_BASE)
else
	PYTHON = $(PYTHON_BASE)
endif

# if you want different behavior for different python versions
# --------------------------------------------------
# COMPATIBILITY_MODE := $(shell $(PYTHON) -c "import sys; print(1 if sys.version_info < (3, 10) else 0)")

# options we might want to pass to pytest
# --------------------------------------------------

# base options for pytest, will be appended to if `COV` or `VERBOSE` are 1.
# user can also set this when running make to add more options
PYTEST_OPTIONS ?=

# set to `1` to run pytest with `--cov=.` to get coverage reports in a `.coverage` file
COV ?= 1
# set to `1` to run pytest with `--verbose`
VERBOSE ?= 0

ifeq ($(VERBOSE),1)
	PYTEST_OPTIONS += --verbose
endif

ifeq ($(COV),1)
	PYTEST_OPTIONS += --cov=.
endif

# ==================================================
# default target (help)
# ==================================================

# first/default target is help
.PHONY: default
default: help

# ==================================================
# getting version info
# we do this in a separate target because it takes a bit of time
# ==================================================

# this recipe is weird. we need it because:
# - a one liner for getting the version with toml is unwieldy, and using regex is fragile
# - using $$GET_VERSION_SCRIPT within $(shell ...) doesn't work because of escaping issues
# - trying to write to the file inside the `gen-version-info` recipe doesn't work, 
# 	shell eval happens before our `python -c ...` gets run and `cat` doesn't see the new file
.PHONY: write-proj-version
write-proj-version:
	@mkdir -p $(VERSIONS_DIR)
	@$(PYTHON) -c "$$GET_VERSION_SCRIPT" > $(VERSION_FILE)

# gets version info from $(PYPROJECT), last version from $(LAST_VERSION_FILE), and python version
# uses just `python` for everything except getting the python version. no echo here, because this is "private"
.PHONY: gen-version-info
gen-version-info: write-proj-version
	@mkdir -p $(LOCAL_DIR)
	$(eval VERSION := $(shell cat $(VERSION_FILE)) )
	$(eval LAST_VERSION := $(shell [ -f $(LAST_VERSION_FILE) ] && cat $(LAST_VERSION_FILE) || echo NULL) )
	$(eval PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')") )

# getting commit log since the tag specified in $(LAST_VERSION_FILE)
# will write to $(COMMIT_LOG_FILE)
# when publishing, the contents of $(COMMIT_LOG_FILE) will be used as the tag description (but can be edited during the process)
# no echo here, because this is "private"
.PHONY: gen-commit-log
gen-commit-log: gen-version-info
	@if [ "$(LAST_VERSION)" = "NULL" ]; then \
		echo "!!! ERROR !!!"; \
		echo "LAST_VERSION is NULL, cant get commit log!"; \
		exit 1; \
	fi
	@mkdir -p $(LOCAL_DIR)
	@$(PYTHON) -c "$$GET_COMMIT_LOG_SCRIPT" "$(LAST_VERSION)"


# force the version info to be read, printing it out
# also force the commit log to be generated, and cat it out
.PHONY: version
version: gen-commit-log
	@echo "Current version is $(VERSION), last auto-uploaded version is $(LAST_VERSION)"
	@echo "Commit log since last version from '$(COMMIT_LOG_FILE)':"
	@cat $(COMMIT_LOG_FILE)
	@echo ""
	@if [ "$(VERSION)" = "$(LAST_VERSION)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Python package $(VERSION) is the same as last published version $(LAST_VERSION), exiting!"; \
		exit 1; \
	fi


# ==================================================
# dependencies and setup
# ==================================================

.PHONY: setup
setup: dep-check
	@echo "install and update via uv"
	@echo "To activate the virtual environment, run one of:"
	@echo "  source .venv/bin/activate"
	@echo "  source .venv/Scripts/activate"

.PHONY: get-cuda-info
get-cuda-info:
	$(eval CUDA_PRESENT := $(shell if command -v nvcc > /dev/null 2>&1; then echo 1; else echo 0; fi))
	$(eval CUDA_VERSION := $(if $(filter $(CUDA_PRESENT),1),$(shell nvcc --version 2>/dev/null | grep "release" | awk '{print $$5}' | sed 's/,//'),NULL))
	$(eval CUDA_VERSION_SHORT := $(if $(filter $(CUDA_PRESENT),1),$(shell echo $(CUDA_VERSION) | sed 's/\.//'),NULL))

.PHONY: dep-check-torch
dep-check-torch:
	@echo "see if torch is installed, and which CUDA version and devices it sees"
	$(PYTHON) -c "$$CHECK_TORCH_SCRIPT"

.PHONY: dep
dep: get-cuda-info
	@echo "Exporting dependencies as per $(PYPROJECT) section 'tool.uv-exports.exports'"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION) | sh -x
	
	@if [ "$(CUDA_PRESENT)" = "1" ]; then \
		echo "CUDA is present, installing torch with CUDA $(CUDA_VERSION)"; \
		uv pip install torch --upgrade --index https://download.pytorch.org/whl/cu$(CUDA_VERSION_SHORT); \
	fi
	

.PHONY: dep-check
dep-check:
	@echo "Checking that exported requirements are up to date"
	uv sync --all-extras --all-groups
	mkdir -p $(REQ_LOCATION)-TEMP
	$(PYTHON) -c "$$EXPORT_SCRIPT" $(PYPROJECT) $(REQ_LOCATION)-TEMP | sh -x
	diff -r $(REQ_LOCATION)-TEMP $(REQ_LOCATION)
	rm -rf $(REQ_LOCATION)-TEMP


.PHONY: dep-clean
dep-clean:
	@echo "clean up lock files, .venv, and requirements files"
	rm -rf .venv
	rm -rf uv.lock
	rm -rf $(REQ_LOCATION)/*.txt

# ==================================================
# checks (formatting/linting, typing, tests)
# ==================================================

# runs ruff and pycln to format the code
.PHONY: format
format:
	@echo "format the source code"
	$(PYTHON) -m ruff format --config $(PYPROJECT) .
	$(PYTHON) -m ruff check --fix --config $(PYPROJECT) .
	$(PYTHON) -m pycln --config $(PYPROJECT) --all .

# runs ruff and pycln to check if the code is formatted correctly
.PHONY: format-check
format-check:
	@echo "check if the source code is formatted correctly"
	$(PYTHON) -m ruff check --config $(PYPROJECT) .
	$(PYTHON) -m pycln --check --config $(PYPROJECT) .

# runs type checks with mypy
# at some point, need to add back --check-untyped-defs to mypy call
# but it complains when we specify arguments by keyword where positional is fine
# not sure how to fix this
.PHONY: typing
typing: clean
	@echo "running type checks"
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(PACKAGE_NAME)/
	$(PYTHON) -m mypy --config-file $(PYPROJECT) $(TYPECHECK_ARGS) $(TESTS_DIR)/

.PHONY: test
test: clean
	@echo "running tests"
	$(PYTHON) -m pytest $(PYTEST_OPTIONS) $(TESTS_DIR)

.PHONY: check
check: clean format-check test typing
	@echo "run format checks, tests, and typing checks"

# ==================================================
# coverage & docs
# ==================================================

# generates a whole tree of documentation in html format.
# see `docs/make_docs.py` and the templates in `docs/templates/html/` for more info
.PHONY: docs-html
docs-html:
	@echo "generate html docs"
	$(PYTHON) docs/make_docs.py

# instead of a whole website, generates a single markdown file with all docs using the templates in `docs/templates/markdown/`.
# this is useful if you want to have a copy that you can grep/search, but those docs are much messier.
# docs-combined will use pandoc to convert them to other formats.
.PHONY: docs-md
docs-md:
	@echo "generate combined (single-file) docs in markdown"
	mkdir $(DOCS_DIR)/combined -p
	$(PYTHON) docs/make_docs.py --combined

# after running docs-md, this will convert the combined markdown file to other formats:
# gfm (github-flavored markdown), plain text, and html
# requires pandoc in path, pointed to by $(PANDOC)
# pdf output would be nice but requires other deps
.PHONY: docs-combined
docs-combined: docs-md
	@echo "generate combined (single-file) docs in markdown and convert to other formats"
	@echo "requires pandoc in path"
	$(PANDOC) -f markdown -t gfm $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME)_gfm.md
	$(PANDOC) -f markdown -t plain $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).txt
	$(PANDOC) -f markdown -t html $(DOCS_DIR)/combined/$(PACKAGE_NAME).md -o $(DOCS_DIR)/combined/$(PACKAGE_NAME).html

# generates coverage reports as html and text with `pytest-cov`, and a badge with `coverage-badge`
# if `.coverage` is not found, will run tests first
# also removes the `.gitignore` file that `coverage html` creates, since we count that as part of the docs
.PHONY: cov
cov:
	@echo "generate coverage reports"
	@if [ ! -f .coverage ]; then \
		echo ".coverage not found, running tests first..."; \
		$(MAKE) test; \
	fi
	mkdir $(COVERAGE_REPORTS_DIR) -p
	$(PYTHON) -m coverage report -m > $(COVERAGE_REPORTS_DIR)/coverage.txt
	$(PYTHON) -m coverage_badge -f -o $(COVERAGE_REPORTS_DIR)/coverage.svg
	$(PYTHON) -m coverage html --directory=$(COVERAGE_REPORTS_DIR)/html/
	rm -rf $(COVERAGE_REPORTS_DIR)/html/.gitignore

# runs the coverage report, then the docs, then the combined docs
.PHONY: docs
docs: cov docs-html docs-combined
	@echo "generate all documentation and coverage reports"

# removed all generated documentation files, but leaves the templates and the `docs/make_docs.py` script
# distinct from `make clean`
.PHONY: docs-clean
docs-clean:
	@echo "remove generated docs"
	rm -rf $(DOCS_DIR)/combined/
	rm -rf $(DOCS_DIR)/$(PACKAGE_NAME)/
	rm -rf $(COVERAGE_REPORTS_DIR)/
	rm $(DOCS_DIR)/$(PACKAGE_NAME).html
	rm $(DOCS_DIR)/index.html
	rm $(DOCS_DIR)/search.js
	rm $(DOCS_DIR)/package_map.dot
	rm $(DOCS_DIR)/package_map.html


# ==================================================
# build and publish
# ==================================================

# verifies that the current branch is $(PUBLISH_BRANCH) and that git is clean
# used before publishing
.PHONY: verify-git
verify-git: 
	@echo "checking git status"
	if [ "$(shell git branch --show-current)" != $(PUBLISH_BRANCH) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not on the $(PUBLISH_BRANCH) branch, exiting!"; \
		exit 1; \
	fi; \
	if [ -n "$(shell git status --porcelain)" ]; then \
		echo "!!! ERROR !!!"; \
		echo "Git is not clean, exiting!"; \
		exit 1; \
	fi; \


.PHONY: build
build: 
	@echo "build the package"
	uv build

# gets the commit log, checks everything, builds, and then publishes with twine
# will ask the user to confirm the new version number (and this allows for editing the tag info)
# will also print the contents of $(PYPI_TOKEN_FILE) to the console for the user to copy and paste in when prompted by twine
.PHONY: publish
publish: gen-commit-log check build verify-git version gen-version-info
	@echo "run all checks, build, and then publish"

	@echo "Enter the new version number if you want to upload to pypi and create a new tag"
	@echo "Now would also be the time to edit $(COMMIT_LOG_FILE), as that will be used as the tag description"
	@read -p "Confirm: " NEW_VERSION; \
	if [ "$$NEW_VERSION" = $(VERSION) ]; then \
		echo "!!! ERROR !!!"; \
		echo "Version confirmed. Proceeding with publish."; \
	else \
		echo "Version mismatch, exiting: you gave $$NEW_VERSION but expected $(VERSION)"; \
		exit 1; \
	fi;

	@echo "pypi username: __token__"
	@echo "pypi token from '$(PYPI_TOKEN_FILE)' :"
	echo $$(cat $(PYPI_TOKEN_FILE))

	echo "Uploading!"; \
	echo $(VERSION) > $(LAST_VERSION_FILE); \
	git add $(LAST_VERSION_FILE); \
	git commit -m "Auto update to $(VERSION)"; \
	git tag -a $(VERSION) -F $(COMMIT_LOG_FILE); \
	git push origin $(VERSION); \
	twine upload dist/* --verbose

# ==================================================
# cleanup of temp files
# ==================================================

# cleans up temp files from formatter, type checking, tests, coverage
# removes all built files
# removes $(TESTS_TEMP_DIR) to remove temporary test files
# recursively removes all `__pycache__` directories and `*.pyc` or `*.pyo` files
# distinct from `make docs-clean`, which only removes generated documentation files
.PHONY: clean
clean:
	@echo "clean up temporary files"
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf $(PACKAGE_NAME).egg-info
	rm -rf $(TESTS_TEMP_DIR)
	$(PYTHON_BASE) -Bc "import pathlib; [p.unlink() for path in ['$(PACKAGE_NAME)', '$(TESTS_DIR)', '$(DOCS_DIR)'] for pattern in ['*.py[co]', '__pycache__/*'] for p in pathlib.Path(path).rglob(pattern)]"

.PHONY: clean-all
clean-all: clean dep-clean docs-clean
	@echo "clean up all temporary files, dep files, venv, and generated docs"


# ==================================================
# smart help command
# ==================================================

# listing targets is from stackoverflow
# https://stackoverflow.com/questions/4219255/how-do-you-get-the-list-of-targets-in-a-makefile
# no .PHONY because this will only be run before `make help`
# it's a separate command because getting the versions takes a bit of time
help-targets:
	@echo -n "# make targets"
	@echo ":"
	@cat Makefile | sed -n '/^\.PHONY: / h; /\(^\t@*echo\|^\t:\)/ {H; x; /PHONY/ s/.PHONY: \(.*\)\n.*"\(.*\)"/    make \1\t\2/p; d; x}'| sort -k2,2 |expand -t 30


.PHONY: info
info: gen-version-info get-cuda-info
	@echo "# makefile variables"
	@echo "    PYTHON = $(PYTHON)"
	@echo "    PYTHON_VERSION = $(PYTHON_VERSION)"
	@echo "    PACKAGE_NAME = $(PACKAGE_NAME)"
	@echo "    VERSION = $(VERSION)"
	@echo "    LAST_VERSION = $(LAST_VERSION)"
	@echo "    PYTEST_OPTIONS = $(PYTEST_OPTIONS)"
	@echo "    CUDA_PRESENT = $(CUDA_PRESENT)"
	@if [ "$(CUDA_PRESENT)" = "1" ]; then \
		echo "    CUDA_VERSION = $(CUDA_VERSION)"; \
		echo "    CUDA_VERSION_SHORT = $(CUDA_VERSION_SHORT)"; \
	fi

.PHONY: info-long
info-long: info
	@echo "# other variables"
	@echo "    PUBLISH_BRANCH = $(PUBLISH_BRANCH)"
	@echo "    DOCS_DIR = $(DOCS_DIR)"
	@echo "    COVERAGE_REPORTS_DIR = $(COVERAGE_REPORTS_DIR)"
	@echo "    TESTS_DIR = $(TESTS_DIR)"
	@echo "    TESTS_TEMP_DIR = $(TESTS_TEMP_DIR)"
	@echo "    PYPROJECT = $(PYPROJECT)"
	@echo "    REQ_LOCATION = $(REQ_LOCATION)"
	@echo "    REQ_BASE = $(REQ_BASE)"
	@echo "    REQ_EXTRAS = $(REQ_EXTRAS)"
	@echo "    REQ_DEV = $(REQ_DEV)"
	@echo "    REQ_ALL = $(REQ_ALL)"
	@echo "    LOCAL_DIR = $(LOCAL_DIR)"
	@echo "    PYPI_TOKEN_FILE = $(PYPI_TOKEN_FILE)"
	@echo "    LAST_VERSION_FILE = $(LAST_VERSION_FILE)"
	@echo "    PYTHON_BASE = $(PYTHON_BASE)"
	@echo "    COMMIT_LOG_FILE = $(COMMIT_LOG_FILE)"
	@echo "    PANDOC = $(PANDOC)"
	@echo "    COV = $(COV)"
	@echo "    VERBOSE = $(VERBOSE)"
	@echo "    RUN_GLOBAL = $(RUN_GLOBAL)"
	@echo "    TYPECHECK_ARGS = $(TYPECHECK_ARGS)"

# immediately print out the help targets, and then local variables (but those take a bit longer)
.PHONY: help
help: help-targets info
	@echo -n ""

# ==================================================
# custom targets
# ==================================================
# (put them down here, or delimit with ~~~~~)