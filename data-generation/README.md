# Synthetic data generation

When it comes to evaluating an LLM application, creating a synthetic test dataset offers a number of advantages. Ragas for example states, that:
> *"human-generated questions may struggle to reach the level of complexity required for a thorough evaluation, ultimately impacting the quality of the assessment"*

[*Synthetic Test Data generation*, [ragas](https://docs.ragas.io/en/stable/concepts/testset_generation.html#why-synthetic-test-data)]

To create a competetive synthetic dataset, these frameworks use data evolution methods to generate a dataset across various complexity levels. This makes the dataset more diverse and challenging, which is crucial for a comprehensive evaluation. The data evolution method was introduced by the the [Evol-Instruct and WizardLM](https://arxiv.org/pdf/2304.12244) paper.

For more detail, read the paper or go through the [ragas documentation](https://docs.ragas.io/en/stable/concepts/testset_generation.html#how-does-ragas-differ-in-test-data-generation) or [deepeval blog](https://www.confident-ai.com/blog/the-definitive-guide-to-synthetic-data-generation-using-llms), how they implemented the data evolution method.

Currently, the synthetic data generation is not implemented completely. A few tests were made, but no definitive results were achieved. For simple Q&A tasks, the *out-of-the-box* capabilities of these frameworks are great, but for more complex tasks, such as formula or workflow creation, a manual and more sophisticated approach needs to be implemented. Test implementations can be found in the ``data_generation_tests.ipynb`` notebook.
