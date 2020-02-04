# Autodoclass: The Automatic Document Classifier

`Autodoclass` aims at being a Automatic Document Classifier by creating
generalized text encoding given a corpus of documents.

The goal being to use a completely unsupervised methodology to automatically
recognize usable metadata which can be used for downstream tasks such as
information extraction, text analysis and search.

## Overview

Many times the initial part of text classification tasks involves a corpus of
documents which need to be classified on multiple levels, such as:

1. Document Type Classification
2. Subsections Classification

Many times the document type and subsections quantities are not known, but the
context is somewhat repeatable. Writing rules can be very time consuming, and
just knowing a heuristic for document type and subsection can be useful enough
to prune texts for information extraction and other downstream tasks.

### Encoding Text

Text encoding can be done in many ways, and some meta data about formatting
could also potentially be useful. At this time, simple text input is supported
for text encoding. All other information is not used or disregarded.

Doc2Vec is the default text encoding mechanism using Spacy's default tokenizer.
Language support for the tokenization can be configured on a per language basis

    Raw Text Input -> Tokenization -> Doc2Vec Encoding

Two Doc2Vec models are automatically created, one on the document level, and
the other based on the texts separated by newlines.

### Deducing Classes

Using HDBSCAN we automatically deduce the amount of classes using well
researched techniques for clustering. The assumption is that given a good
enough amount of data, we can determine similarity deducing relatedness hence
a finite class set.

## Usage

Basic usage, training, predicting, saving and loading.

```
>>> adc = Autodoclass()
>>> adc.train("/path/to/text/files")
>>> adc.predict("Some raw text data\nWith multiple lines")
( (10, 0.7), ((1, 0.75), ... (12, 0.87)) )
>>> adc.save("adc.model.date.pkl")
>>> Autodoclass.load("adc.model.date.pkl")
```
