.PHONY: clean test lint

help:
	@echo "	clean"
	@echo "		Remove Python/build artifacts."
	@echo "	formatter"
	@echo "		Apply black formatting to code."
	@echo "	lint"
	@echo "		Lint code with flake8, and check if black formatter should be applied."
	@echo "	types"
	@echo "		Check for type errors using pytype."
	@echo "	readme-toc"
	@echo "		Generate a Table Of Content for the README.md"
	@echo "	build-docker-actions"
	@echo "		Build Docker image for the Action Server"
	@echo "	build-docker-actions"
	@echo "		Build Docker image for the Action Server"
	@echo "	tag"
	@echo "		Create a git tag based on the current pacakge version and push"


clean:
	find . -name 'README.md.*' -exec rm -f  {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f  {} +
	rm -rf build/
	rm -rf .pytype/
	rm -rf dist/
	rm -rf docs/_build

formatter:
	black .

lint:
	flake8 *_engine tests bot_actions
	black --check .

types:
	pytype --keep-going .

readme-toc:
	# Regenerates all Readme's Table of Contents
	# https://github.com/ekalinin/github-markdown-toc
	find . -name README.md -exec gh-md-toc --insert {} \;

build-docker-actions:
	# Examples:
	# make build-docker-actions mode=dev version=0.1
	# make build-docker-actions mode=full version=0.1
	docker build --rm \
		-f Dockerfile \
		-t wiki_qa_demo:$(version) \
		-t wiki_qa_demo:latest \
		bot_actions

tag:
	# use as 'make tag version=0.1
	git tag $(version)
	git push --tags
