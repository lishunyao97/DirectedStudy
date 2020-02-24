# DirectedStudy

## Automatic Entity-based Distractor Generation

SmartReader System requires a time-efficient, high quality distractor generator to test readers’ understanding about English passages. This project proposed a novel way to generate entity-based distractors based on fine-grained question classification and BERT embeddings. The automatic entity-based distractor generation system can be initialized in 10 seconds and the averaging time consumed to generate three distractors for a question-answer pair is 2.3s. The system also beats baseline model on overall, grammar and semantics score in human evaluation.

## requirements

- python>=3.5
- numpy>=1.18.1
- spacy>=2.2.3
- download spacy pretrained model: python -m spacy download en_core_web_lg
- pywikibot>=3.0.20200111
- nltk>=3.4.5
- tensorflow>=1.10 (as a prerequisite for bert-as-service)
- [bert-as-service](https://github.com/hanxiao/bert-as-service)


