# gds-document-clustering

Feasibility of using machine classification to improve GOV.UK taxonomy

## Setup

* Install the version of Python specified in `.python-version`, e.g. `mise install python` (with `idiomatic_version_file_enable_tools` enabled for Python).
* Install `pipenv` by running `pip install --user pipenv`.
* Install Python libraries by running `pipenv install`.
* Sign up at [Hugging Face](https://huggingface.co/) and create a token of type "Read".
* Copy example environment file: `cp .env.example .env`.
* Set value of `HF_TOKEN` in `.env` to the Hugging Face token.
