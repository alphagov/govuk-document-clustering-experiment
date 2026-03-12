# GOV.UK Document Clustering Experiment

Feasibility of using machine classification to improve GOV.UK taxonomy

## Setup

* [Setup a GOV.UK development environment](https://docs.publishing.service.gov.uk/manual/get-started.html).
* Setup the [Content Tagger application](https://docs.publishing.service.gov.uk/repos/content-tagger.html) so it can be run from [govuk-docker](https://docs.publishing.service.gov.uk/repos/govuk-docker.html).
* [Replicate](https://docs.publishing.service.gov.uk/repos/govuk-docker/how-tos.html#how-to-replicate-data-locally) the production data for Content Tagger locally.
* Install the version of Python specified in `.python-version`, e.g. `mise install python` (with `idiomatic_version_file_enable_tools` enabled for Python).
* Install `pipenv` by running `pip install --user pipenv`.
* Install Python libraries by running `pipenv install`.
* Sign up at [Hugging Face](https://huggingface.co/) and create a token of type "Read".
* Copy example environment file: `cp .env.example .env`.
* Set value of `HF_TOKEN` in `.env` to the Hugging Face token.
* Set the value of `OPENROUTER_API_KEY` in `.env` to the [Open Router](https://openrouter.ai) API key.
* Generate the suggested topics: `pipenv run ./suggest_topics <taxon-base-path>`
