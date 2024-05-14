# RAG evaluation prototypes

This repository contains prototypes to evaluate RAG pipelines, as well as some results.

The jupyter notbooks contain code to generate synthetic test and evaluation data based on the ragas docs (with additional adjustments in ``custom_evolutions.py``.

TODO:

- [ ] seperate synthetic data generation and evaluation logic (Dockerfile for evaluation endpoint)
- [ ] add different evaluation endpoints: traditional Q&A and specific tasks (e.g. Formula or Workflow creation)
- [ ] optional: try to include DevExpress formula evaluation (just a quick check, if the formula is syntactically correct)
