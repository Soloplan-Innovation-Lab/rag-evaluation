from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
from ragas.testset.evolutions import Evolution, EvolutionOutput, CurrentNodes
from ragas.llms.prompt import Prompt
from ragas.testset.utils import rng

logger = logging.getLogger(__name__)

# these parts are based on ragas and the evol-instruct paper
# ref: https://docs.ragas.io/en/latest/concepts/testset_generation.html#how-does-ragas-differ-in-test-data-generation
# ref: https://arxiv.org/pdf/2304.12244

@dataclass
class FormulaEvolution(Evolution):
    formula_question_prompt: Prompt = field(default_factory=lambda: formula_question_prompt)

    async def _aevolve(
        self, current_tries: int, current_nodes: CurrentNodes
    ) -> EvolutionOutput:
        assert self.docstore is not None, "docstore cannot be None"
        assert self.node_filter is not None, "node filter cannot be None"
        assert self.generator_llm is not None, "generator_llm cannot be None"
        assert self.question_filter is not None, "question_filter cannot be None"

        merged_node = self.merge_nodes(current_nodes)
        passed = await self.node_filter.filter(merged_node)
        if not passed["score"]:
            current_nodes = self._get_new_random_node()
            return await self.aretry_evolve(current_tries, current_nodes, update_count=False)

        logger.debug("Merged node content: %s", merged_node.page_content)
        results = await self.generator_llm.generate(
            prompt=self.formula_question_prompt.format(
                context=merged_node.page_content,
                question=rng.choice(np.array(merged_node.keyphrases), size=1)[0]
            )
        )
        generated_question = results.generations[0][0].text
        logger.info("Generated question: %s", generated_question)
        is_valid_question, feedback = await self.question_filter.filter(generated_question)

        if not is_valid_question:
            generated_question, current_nodes = await self.fix_invalid_question(
                generated_question, current_nodes, feedback
            )
            logger.info("Rewritten question: %s", generated_question)
            is_valid_question, _ = await self.question_filter.filter(generated_question)
            if not is_valid_question:
                current_nodes = self._get_new_random_node()
                return await self.aretry_evolve(current_tries, current_nodes)

        return generated_question, current_nodes, "formula"

    def __hash__(self):
        return hash(self.__class__.__name__)

    def adapt(self, language: str, cache_dir: Optional[str] = None) -> None:
        super().adapt(language, cache_dir)
        self.formula_question_prompt = self.formula_question_prompt.adapt(
            language, self.generator_llm, cache_dir
        )

    def save(self, cache_dir: Optional[str] = None) -> None:
        super().save(cache_dir)
        self.formula_question_prompt.save(cache_dir)

formula_question_prompt = Prompt(
    name="formula_question",
    instruction="""Rewrite the given prompt to create a question that requires generating a complete and correctly formatted DevExpress formula based on the provided context.
    Ensure the question is specific to DevExpress Criteria Language and can be answered using the context.
    Rules to follow when rewriting the question:
    1. Ensure that the rewritten question can be answered entirely from the information present in the contexts.
    2. Do not frame questions that contain more than 15 words. Use abbreviations wherever possible.
    3. Make sure the question is clear and unambiguous.
    4. Phrases like 'based on the provided context', 'according to the context', etc., are not allowed to appear in the question.
    5. Ensure the answer is a single, complete, and correctly formatted DevExpress formula.""",
    examples=[
        {
            "question": "Create a formula that checks whether the end date is still in the current month when 5 working days are added to the current date.",
            "context": "FunctionName: [IsThisMonth(DateTime)] Description: [Returns True if the specified date falls within the current month.] Example: [IsThisMonth([OrderDate])], FunctionName: [Today()] Description: [Returns the current date. Regardless of the actual time, this function returns midnight of the current date.] Example: [AddMonths(Today(), 1)], FunctionName: [AddWorkingDays] Description: [AddWorkingDays(DateTime, DaysCount, [optional]Iso2Code,[optional]ZipCode) This function adds a number of working days (DaysCount) to the start date (DateTime) and returns a date-time value. If required, holidays can be taken into consideration. The following parameters are available: DateTime = Original date-time value DaysCount = Number of working days Iso2Code = Country for which the holidays are to be considered (optional) ZipCode = Additional postcode so that local holidays can be considered (optional)]",
            "output": "IsThisMonth(AddWorkingDays(Today(), 5))"
        },
        {
            "question": "Write a DevExpress expression to calculate the total of a field named 'Sales' for records where 'Region' is 'North'.",
            "context": "FunctionName: [Sum(Value)] Description: [Returns the sum of all the expression values in the collection.] Example: [[Products].Sum([UnitsInStock])], FunctionName: [Iif(Expression1, True_Value1, ..., ExpressionN, True_ValueN, False_Value)] Description: [Returns one of several specified values depending upon the values of logical expressions. The function can take 2N+1 arguments (N - the number of specified logical expressions).",
            "output": "Sum(Iif([Region] == 'North', [Sales], Null))"
        },
        {
            "question": "Create a DevExpress Criteria Language expression to filter records where the 'City' is in 'New York', 'Los Angeles', 'Chicago'.",
            "context": "FunctionName: [In(Expression, Value1, Value2, ...)] Description: [Returns True if the expression value is found within the specified list of values.] Example: [[City] In ('New York', 'Los Angeles', 'Chicago')]",
            "output": "[City] In ('New York', 'Los Angeles', 'Chicago')"
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="english",
)

formula_evolution = FormulaEvolution()
