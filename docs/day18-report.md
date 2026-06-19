# Day 18 Report — NLP Pipeline & Text Intelligence

## Technical Summary

Today I implemented a Natural Language Processing (NLP) pipeline to process unstructured user text data. The pipeline included text cleaning, feature extraction using TF-IDF, sentiment analysis, and interest mapping.

## Implementation

* Developed a text cleaning function using tokenization and stopword removal
* Converted cleaned text into numerical representation using TF-IDF Vectorizer
* Extracted key user interests by identifying top TF-IDF weighted words
* Performed sentiment analysis using TextBlob to determine polarity and subjectivity

## Bug Log

**Issue 1:** LookupError for 'punkt_tab' in NLTK
**Cause:** Missing tokenizer resource required by the updated NLTK version
**Solution:** Installed the missing resource using `nltk.download('punkt_tab')`

**Issue 2:** ModuleNotFoundError for nltk in Jupyter
**Cause:** Jupyter Notebook was using a different Python environment instead of the project virtual environment
**Solution:** Linked the virtual environment to Jupyter using `ipykernel` and switched the kernel

**Issue 3:** SyntaxError while running ipykernel command
**Cause:** Attempted to run a terminal command inside a Python cell
**Solution:** Used `!` prefix in Jupyter or ran the command in terminal

## Key Insight

TF-IDF helped identify the most significant words that define a user’s interests, while sentiment analysis provided insight into the emotional tone of user-generated content.

## Conceptual Understanding

Text preprocessing is essential because raw text contains noise such as punctuation and common words that do not contribute meaningful information. TF-IDF transforms text into a structured numerical format that models can understand.

## Reflection

Sentiment analysis can help platforms like MeetMux automatically detect negative user feedback. If users consistently express negative sentiment in reviews, the system can flag those events or organizers for further investigation, improving overall platform quality and user experience.

## Additional Learning

Lemmatization reduces words to their root form while preserving meaning, whereas stemming simply cuts words, which may lead to loss of context. Lemmatization is more accurate for NLP pipelines.
