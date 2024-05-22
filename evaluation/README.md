# Evaluation

For the RAG pipeline (or LLM application) evaluation, [deepeval](https://github.com/confident-ai/deepeval) and [ragas](https://github.com/explodinggradients/ragas) are currently used. Another alternative to use could be [ARES](https://github.com/stanford-futuredata/ARES). Note, that currently a lot of metrics are included, also duplicate ones. After running many tests, the selection of metrics and their impact on the results can be evaluted and adapted (as well as the evaluation framework itself). Currently, the following metrics are used:

<ins>**deepeval:**</ins>

- **AnswerRelevancyMetric**: evaluates whether the prompt template in your generator is able to instruct your LLM to output relevant and helpful outputs based on the retrieval_context.
- **FaithfulnessMetric**: evaluates whether the LLM used in your generator can output information that does not hallucinate AND contradict any factual information presented in the retrieval_context.
- **ContextualPrecisionMetric**: evaluates whether the reranker in your retriever ranks more relevant nodes in your retrieval context higher than irrelevant ones.
- **ContextualRecallMetric**: evaluates whether the embedding model in your retriever is able to accurately capture and retrieve relevant information based on the context of the input.
- **ContextualRelevancyMetric**: evaluates whether the text chunk size and top-K of your retriever is able to retrieve information without much irrelevancies.

<ins>**ragas:**</ins>

- **faithfulness**: measures the factual consistency of the generated answer against the given context.
- **answer_correctness**: the assessment involves gauging the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the ground truth and the answer.
- **answer_relevancy**: focuses on assessing how pertinent the generated answer is to the given prompt..
- **context_precision**: evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not.
- **context_recall**: measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth.
- **context_entity_recall**: gives the measure of recall of the retrieved context, based on the number of entities present in both ground_truths and contexts relative to the number of entities present in the ground_truths alone.
- **answer_similarity**: assessment of the semantic resemblance between the generated answer and the ground truth.

*For a more in-depth overview on how these metrics are calculated, please check out the [deepeval](https://docs.confident-ai.com/docs/metrics-introduction) and [ragas](https://docs.ragas.io/en/stable/concepts/metrics/index.html) metrics documentation.*

### Specific evaluation metrics

It is worth mentioning, that the previous listed metrices are mostly generic and simply based on math and statistics. This means, that the semantic similarities are calculated by comparing the embeddings of the text values. Also, they do not capture domain-specific requirements deeply. For more domain-specific metrics, approaches like GEval ([deepeval docs](https://docs.confident-ai.com/docs/metrics-llm-evals), [paper](https://arxiv.org/pdf/2303.16634)) or SemScore ([paper](https://arxiv.org/pdf/2401.17072)) can be used.

Other specific metrics, e.g. for the formula creation (like the formula syntax, length, ...), custom metrics (like previously mentioned) can be implemented and tested. For syntax checking of formulas, the [CriteriaOperator.TryParse](https://supportcenter.devexpress.com/ticket/details/t812826/how-to-calculate-a-criteria-and-check-its-syntax-at-runtime) could be considered.

### How does the evaluation work?

In order to evaluate the RAG pipeline effectively, it is good to have a predefined dataset, that contains the following information:

- **input**: the question that is asked
- **context**: the ideal retrieval results for a given input
- **expected_output**: the expected output from the model

To evaluate the real results of the RAG pipeline or the LLM application, the ``input`` is sent to the application. The application retrieves the ``retrieval_context`` to generate the ``actual_output``. These values are then aggregated in the existing dataset and the whole dataset is then evaluated. The metrics are then calculated by comparing these results. The similarity between the ``expected_output`` and the ``actual_output`` is measured, as well as the relevance of the ``retrieval_context`` to the ``input`` or if the ``input`` is relevant to the ``expected_output``.

Since some metrics use LLMs to create the results, a number of evaluation iterations can be provided (*``iterations_per_entry`` in the evaluation request*) in order to run these metrics multiple times. This should lead to more stable results, by averaging the results of multiple runs. Note, that all iterations are stored in the database, thus statistics like mean and standard deviation can be calculated.

### How to run the evaluation?

If the docker image is not running, you can start it by running the following command:

```bash
docker-compose up -d --build
```

This does build the ``Dockerfile``, which creates the application by copying all relevant files into the image. Also, the [MongoDB](https://github.com/mongodb/mongo) database is started, which is used to store the evaluation results.

By accessing [localhost:8000/docs](http://localhost:8000/docs), the swagger documentation displays all available endpoints. To start an evaluation, the data can simply be sent to the ``/evaluate`` endpoint. An example request looks like this:

```json
{
  "dataset": [
    {
      "input": "Create a formula that checks whether the end date is still in the current month when 5 working days are added to the current date.",
      "retrieval_context": [
        "FunctionName: [IsThisMonth(DateTime)] Description: [Returns True if the specified date falls within the current month.] Example: [IsThisMonth([OrderDate])]",
        "FunctionName: [Today()] Description: [Returns the current date. Regardless of the actual time, this function returns midnight of the current date.] Example: [AddMonths(Today(), 1)]",
        "FunctionName: [AddWorkingDays] Description: [AddWorkingDays(DateTime, DaysCount, [optional]Iso2Code,[optional]ZipCode) This function adds a number of working days (DaysCount) to the start date (DateTime) and returns a date-time value. If required, holidays can be taken into consideration. The following parameters are available: DateTime = Original date-time value DaysCount = Number of working days Iso2Code = Country for which the holidays are to be considered (optional) ZipCode = Additional postcode so that local holidays can be considered (optional)]"
      ],
      "context": null,
      "actual_output": "IIf(AddWorkingDays(Today(), 5), IsThisMonth()) > 0",
      "expected_output": "IsThisMonth(AddWorkingDays(Today(), 5))"
    }
  ],
  "iterations_per_entry": 1,
  "description": "First test!",
  "run_type": "formula"
}
```
