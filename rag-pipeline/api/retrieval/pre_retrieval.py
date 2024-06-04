from abc import ABC, abstractmethod
from typing import List
from langchain_core.messages import AIMessage, BaseMessage
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
)
from llm import invoke_prompt, invoke_prompt_async
from internal_shared.models.chat import PreRetrievalType


class PreRetrievalStrategy(ABC):
    """
    Abstract class for pre-retrieval strategies.
    """

    @abstractmethod
    def execute(self, query: str) -> str:
        """
        Executes the pre-retrieval strategy.
        """
        pass

    @abstractmethod
    async def execute_async(self, query: str) -> str:
        """
        Executes the pre-retrieval strategy asynchronously.
        """
        pass


class DefaultPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Default pre-retrieval strategy. No modifications to the query.
    """

    def execute(self, query: str) -> str:
        return query

    async def execute_async(self, query: str) -> str:
        return query


class QueryExpansionPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Query expansion strategy based on the paper "Query Expansion by Prompting Large Language Models" (https://arxiv.org/pdf/2305.03653)

    Compared to approaches like PRF, query expansion with LLMs is more flexible and can be used to expand queries with more complex semantics. This approach also relies on "general-purpose" LLMs, rather than specialized models for query expansion.
    """

    def _get_messages(self, query: str) -> List:
        return [
            AIMessage(
                f"Answer the following query: {query}\nGive the rationale before answering"
            ),
        ]

    def _format_query(self, query: str, response: str) -> str:
        return (f"{query} " * 5) + response

    def execute(self, query: str) -> str:
        messages = self._get_messages(query)
        response = invoke_prompt(messages)
        return (f"{query} " * 5) + response.content

    async def execute_async(self, query: str) -> str:
        messages = self._get_messages(query)
        response = await invoke_prompt_async(messages)
        return self._format_query(query, response.content)


class RewriteRetrieveReadPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Rewrite-retrieve-read strategy basd on the paper "Query Rewriting for Retrieval-Augmented Large Language Models" (https://arxiv.org/pdf/2305.14283)

    The overall idea is to rewrite the query before the retrieval step. Thanks to approaches like CoT and ReAct, rewriting can enhance the performance of LLMs in information retrieval tasks. Besides using a frozen LLM for rewriting, trainable approach using RL is also proposed in the paper.
    """

    def _get_few_shot_examples(self) -> List:
        ex_1 = (
            "Question: '''Rita Coolidge sang the title song for which Bond film?'''"
            "Thought: I need to find information about Bond films and songs associated with Rita Coolidge to identify the correct Bond film."
            "Query: Rita Coolidge Bond title song; Bond film title song Rita Coolidge; Bond movies songs Rita Coolidge**"
        )
        return [ex_1]

    def _get_messages(self, query: str, use_few_shot: bool) -> List:
        if use_few_shot:
            fs_examples = "\n".join(self._get_few_shot_examples())
            return [
                AIMessage(
                    f"Think step by step to answer this question, and provide queries optimized for vector databases to retrieve the necessary knowledge. Split the queries with ';' and end the queries with '**'. "
                    + fs_examples
                    + f"Question: {query} Answer:"
                )
            ]
        return [
            AIMessage(
                f"Think step by step to answer this question, and provide queries optimized for vector databases to retrieve the necessary knowledge. Split the queries with ';' and end the queries with '**'. Question: {query} Answer:"
            )
        ]

    def execute(self, query: str) -> str:
        messages = self._get_messages(query, use_few_shot=False)
        response = invoke_prompt(messages)
        return response.content

    async def execute_async(self, query: str) -> str:
        messages = self._get_messages(query, use_few_shot=False)
        response = await invoke_prompt_async(messages)
        return response.content


class StepBackPromptingPreRetrievalStrategy(PreRetrievalStrategy):
    """
    Based on the paper "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models" (https://arxiv.org/pdf/2310.06117)

    Adapted to work with the retrieval pipeline. This part implements "Step 1: Abstraction" and is used to create the stepback question for the upcoming retrieval step.

    Note, that this approach could also improve the performance of the LLM answer generation in general.
    """

    def _get_stepback_message(self, query: str) -> List[BaseMessage]:
        examples = [
            {
                "input": "Could the members of The Police perform lawful arrests?",
                "output": "what can the members of The Police do?",
            },
            {
                "input": "Jan Sindel's was born in what country?",
                "output": "what is Jan Sindel's personal history?",
            },
        ]
        # We now transform these to example messages
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at world knowledge. Your task is to step back "
                    "and paraphrase a question to a more generic step-back question, which "
                    "is easier to answer. Here are a few examples:",
                ),
                # Few shot examples
                few_shot_prompt,
                # New question
                ("user", "{question}"),
            ]
        )

        return prompt.format_messages(question=query)

    def execute(self, query: str) -> str:
        messages = self._get_stepback_message(query)
        return invoke_prompt(messages).content

    async def execute_async(self, query: str) -> str:
        messages = self._get_stepback_message(query)
        result = await invoke_prompt_async(messages)
        return result.content


class HydePreRetrievalStrategy(PreRetrievalStrategy):
    """ "
    Based on the paper "Precise Zero-Shot Dense Retrieval without Relevance Labels" (https://arxiv.org/pdf/2212.10496)

    Based on the existing templates in their repository, a new "HYPOTHETICAL_INTERFACE" template is used for this strategy.
    """

    HYPOTHETICAL_INTERFACE = """  
    Based on the following query, please write a hypothetical C# interface documentation that might answer the query.  
    
    Query: {question}  
    Hypothetical Interface Documentation:  
    """

    def generate_prompt(self, query: str) -> str:
        template = PromptTemplate.from_template(self.HYPOTHETICAL_INTERFACE)
        return template.format(question=query)

    def execute(self, query: str) -> str:
        prompt = self.generate_prompt(query)
        return invoke_prompt(prompt).content

    async def execute_async(self, query: str) -> str:
        prompt = self.generate_prompt(query)
        result = await invoke_prompt_async(prompt)
        return result.content


class RephraseAndRespondePreRetrievalStrategy(PreRetrievalStrategy):
    """
    Based on the paper "Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves" (https://arxiv.org/pdf/2311.04205)

    Besides the normal (One-Step) RaR, they also propose a Two-Step RaR, which is implemented here. The authors also describe few-shot examples for the Two-Step RaR, which improves the performance of the model.

    Note, that this approach is not only for the retrieval step, but could also be used to improve the performance of the LLM answer generation in general.
    """

    ONE_STEP_RAR = """
    {question}
    Rephrase and expand the question, and respond.
    """

    TWO_STEP_RAR = """
    {question}
    Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
    """

    def generate_prompt(self, query: str) -> str:
        template = PromptTemplate.from_template(self.TWO_STEP_RAR)
        return template.format(question=query)

    def execute(self, query: str) -> str:
        prompt = self.generate_prompt(query)
        return invoke_prompt(prompt).content

    async def execute_async(self, query: str) -> str:
        prompt = self.generate_prompt(query)
        result = await invoke_prompt_async(prompt)
        return result.content


class PreRetrievalStrategyFactory:
    """
    Factory class to create pre-retrieval strategies.
    """

    @staticmethod
    def create(strategy_type: PreRetrievalType) -> PreRetrievalStrategy:
        match strategy_type:
            case PreRetrievalType.DEFAULT:
                return DefaultPreRetrievalStrategy()
            case PreRetrievalType.QUERY_EXPANSION:
                return QueryExpansionPreRetrievalStrategy()
            case PreRetrievalType.REWRITE_RETRIEVE_READ:
                return RewriteRetrieveReadPreRetrievalStrategy()
            case PreRetrievalType.STEP_BACK_PROMPTING:
                return StepBackPromptingPreRetrievalStrategy()
            case PreRetrievalType.HYDE:
                return HydePreRetrievalStrategy()
            case PreRetrievalType.REPHRASE_AND_RESPOND:
                return RephraseAndRespondePreRetrievalStrategy()
            case _:
                return DefaultPreRetrievalStrategy()


class PreRetrievalStep:
    """
    Facade class to execute pre-retrieval strategies.
    """

    @staticmethod
    def execute(strategy_type: PreRetrievalType, query: str) -> str:
        """
        Execute a pre-retrieval strategy based on the given configuration.
        """
        pre = PreRetrievalStrategyFactory.create(strategy_type)
        return pre.execute(query)

    @staticmethod
    async def execute_async(strategy_type: PreRetrievalType, query: str) -> str:
        """
        Execute a pre-retrieval strategy based on the given configuration asynchronously.
        """
        pre = PreRetrievalStrategyFactory.create(strategy_type)
        return await pre.execute_async(query)
