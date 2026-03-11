---
layout: default
title: Home
---

## Introduction

This website documents some document clustering experiments that were run against [the GOV.UK Topic Taxonomy](https://docs.publishing.service.gov.uk/manual/taxonomy.html) using the Python [BERTopic library](https://maartengr.github.io/BERTopic/) via the code in [this GitHub repository](https://github.com/freerange/gds-document-clustering).

[The Taxonomy Health page](https://content-tagger.integration.publishing.service.gov.uk/taxonomy/health_warnings) in the [Content Tagger application](https://docs.publishing.service.gov.uk/repos/content-tagger.html) indicates that a significant number of taxons have "too much content tagged to them", i.e. they are associated with [more than 300 content items](https://github.com/alphagov/content-tagger/blob/2f41491c37a6e0f986efdce19da5d64a58db9acd/config/health_checks.yml#L7-L10).

The BERTopic library was used to generate candidate clusters of content items to split up the content items into new child taxons (i.e. sub-topics) of the taxon in question. If a sufficient number of content items could be pushed down into new sub-topics, this would make the topic healthy according to the current heuristic.

## Topic selection

A couple of unhealthy topic taxons with approx 1,000 content items were identified. These were both level 2 taxons "leaf" taxons so that there was obviously "room" for the creation of new sub-topics without exceeding [the maximum depth of 5](https://github.com/alphagov/content-tagger/blob/2f41491c37a6e0f986efdce19da5d64a58db9acd/config/health_checks.yml#L3-L6).

* [Work](https://www.gov.uk/work) > [Labour market reform](https://www.gov.uk/employment/labour-market-reform) (998 content items)
* [Going and being abroad](https://www.gov.uk/going-and-being-abroad) > [Travel abroad](https://www.gov.uk/going-and-being-abroad/travel-abroad) (1,110 content items)

## Common configuration

The BERTopic clustering script was run against the data for these example taxons as follows:
* Using a database dump from the Content Store application from January 2026 as the source of content items.
* A random number generator seed was used to improve reproducibility across runs to make comparisons easier.
* Using the title (repeated twice for emphasis) and the body (stripped of HTML tags) of each content item.
* No content was included from PDF/HTML attachments associated with each conten item.
* English stop words were excluded from the keywords generated for each cluster.
* The keywords for each cluster were constrained to be either one or two words.
* They keywords for each cluster were converted into human-friendly topic names by using the built-in LLM integration (using Open AI's `gpt-4o-mini` model) with 4 example documents for each cluster limited to 2,000 tokens per document.
* The default LLM prompt was [simplified](https://github.com/freerange/gds-document-clustering/commit/aaab3ebc4ff19150ea596cf489b527dad63bc0c3), but the "three words at most" constraint (for the topic name) was initially left in place.
* The documents not in a cluster were listed under an "Outliers" topic.

## Configuration variations

The clustering script was run with the following variations:
1. No topic reduction.
2. [Automatic topic reduction](https://maartengr.github.io/BERTopic/getting_started/topicreduction/topicreduction.html#automatic-topic-reduction).
3. [Manual topic reduction](https://maartengr.github.io/BERTopic/getting_started/topicreduction/topicreduction.html#manual-topic-reduction) with 12 topics specified. 12 topics is the maximum number of child taxons allowed by [another taxonomy health metric](https://github.com/alphagov/content-tagger/blob/2f41491c37a6e0f986efdce19da5d64a58db9acd/config/health_checks.yml#L11-L15). The outliers are counted as a topic, so only 11 topics are actually generated.
4. Manual topic reduction with 6 topics specified. The outliers are counted as a topic, so only 5 topics are actually generated.
5. Manual topic reduction with 12 topics specified, but with "three words at most" in the LLM prompt changed to "five words at most". The average number of words in a topic name across the existing Topic taxonomy is approx 5.
6. Manual topic reduction with 12 topics specified, but with "three words at most" in the LLM prompt changed to "five words at most" and with the following extra context included in the prompt: "The topic already has a parent topic, "$parent-topic-name", which itself has a parent topic, "$grandparent-topic-name". The topic label can assume this context, i.e. it not necessary to repeat these terms in the topic label."

## Results

The results for each of the selected topics can be viewed here:
* [Work > Labour market reform](./employment/labour-market-reform/)
* [Going and being abroad > Travel abroad](./going-and-being-abroad/travel-abroad/)
