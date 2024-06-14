from internal_shared.utils.timer import atime_wrapper
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import LanguageModelInput
from llm import embed_text_async, invoke_prompt_async
from pipeline.helper import (
    calculate_token_usage,
    render_prompt,
)
from retrieval import RetrievalStep
from internal_shared.models.ai.available_models import AvailableModels
from internal_shared.models.chat import (
    ChatResponse,
    PostRetrievalType,
    PreRetrievalType,
    RetrievalConfig,
    RetrievalStepResult,
    RetrieverConfig,
    RetrieverType,
)


class ChatAgent:
    def __init__(self, template: str):
        self.template = template

    async def invoke_prompt_async(
        self,
        prompt: LanguageModelInput,
        chat_model: AvailableModels = AvailableModels.GPT_4O,
    ):
        return await invoke_prompt_async(prompt, chat_model)


class FormulaChatAgent(ChatAgent):
    """
    This agent generates formulas based on the input from the user and the results from the FormulaKnowledgeAgent, as well as the BusinessLogicKnowledgeAgent.
    """
    name: str = "formula_chat_agent"

    TEMPLATE: str = """You are an expert in creating DevExpress Criteria Language Expressions and problem-solving. Generate complex formulas swiftly and accurately based solely on the provided context and your DevExpress knowledge. Respond with the formula ONLY!
  
    To fulfill your task, you are provided with the following context:
    {context}
    
    Examples:
    {examples}
    
    Before reasoning and creating the formula, ensure you understand the provided context. Then, create the formula based on the context provided and comply with the following rules:
    1. Properties must be enclosed in square brackets (e.g., [PropertyName]).
    2. Functions and methods should follow the DevExpress syntax.
    3. Ensure all conditions are correctly formatted and combined.
    4. Pay close attention to the order of parameters in functions and methods.
    5. If no information about optional parameters is given, you can omit them.
    
    Note: Carefully ensure the parameters are placed in the correct order as defined in the function description.  
    
    If there is nothing in the context relevant to the question at hand, just say "I am not sure." and include a reason why you were not able to process the request. Don't try to make up an answer.
    
    Your expertise is required to formulate an accurate response quickly. Make use of the different context sections to create the formula. Respond with the formula ONLY."""

    # validation "step"
    """After generating the formula, review it to ensure:
    1. The formula uses the correct properties and functions as per the provided context.
    2. The parameters are in the correct order.
    3. The conditions are correctly formatted and combined.
    
    If you find any issues during the review, correct them before finalizing your response."""

    FEW_SHOT_EXAMPLES = """User: Create a formula, that checks, if adding 5 working days to the current date, if it is in the current month.
    Assistant: [IsThisMonth(AddWorkingDays(Today(), 5))]

    User: Create a formula, that writes the first designation of the vehicle groups of the truck of the current tour to a string separated by a semicolon.
    Assistant: ItemsToText([ITour.Truck.VehicleGroup], '[First]', '[Designation]', ';')"""

    def __init__(self):
        super().__init__(self.TEMPLATE)

    async def execute_async(
        self,
        query: str,
        context: str,
        chat_id: str,
        chat_model: AvailableModels = AvailableModels.GPT_4O,
    ):
        chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", self.template),
                ("user", "{query}"),
            ]
        )
        prompt = await chat_template.aformat_messages(context=context, query=query, examples=self.FEW_SHOT_EXAMPLES)
        res_time, response = await atime_wrapper(
            self.invoke_prompt_async, prompt, chat_model
        )
        print(f"{self.name}: Response duration: {res_time}")
        token_usage = calculate_token_usage(prompt, response, chat_model.value)
        rendered_prompt = render_prompt(prompt)
        print(f"{self.name}: Token usage: {token_usage}")

        return ChatResponse(
            response=response.content,
            request=query,
            rendered_prompt=rendered_prompt,
            model=chat_model,
            response_duration=res_time,
            token_usage=token_usage,
            chat_session_id=chat_id,
        )


class KnowledgeAgent(ChatAgent):
    def __init__(self, name: str,  template: str, retrieval_config: RetrievalConfig):
        super().__init__(template)
        self.name = name
        self.template = template
        self.behaviour = retrieval_config

    async def execute_async(
        self,
        query: str,
        chat_id: str,
        chat_model: AvailableModels = AvailableModels.GPT_4O,
    ):
        embed_time, embedded_query = await atime_wrapper(
            embed_text_async, query, self.behaviour.retriever.embedding_model
        )
        print(f"{self.name}: Embedding duration: {embed_time}")

        retrieval_time, documents = await atime_wrapper(
            RetrievalStep.execute_async, self.behaviour, embedded_query
        )
        print(f"{self.name}: Retrieval duration: {retrieval_time}")

        step = RetrievalStepResult(
            config=self.behaviour,
            initial_query=query,
            pre_retrieval=query,
            pre_retrieval_duration=0,
            retrieval=documents,
            retrieval_duration=0,
            post_retrieval=documents,
            post_retrieval_duration=0,
        )

        prompt_template = PromptTemplate.from_template(self.template)
        ctx = [doc.content for doc in documents]
        ctx.extend(RetrievalStep.get_curated_documents())
        context_str = "\n".join(ctx)
        prompt = await prompt_template.aformat_prompt(context=context_str, query=query)

        res_time, response = await atime_wrapper(
            self.invoke_prompt_async, prompt, chat_model
        )

        print(f"{self.name}: Response duration: {res_time}")

        token_usage = calculate_token_usage(prompt, response, chat_model.value)

        print(f"{self.name}: Token usage: {token_usage}")

        return ChatResponse(
            chat_session_id=chat_id,
            response=response.content,
            documents=ctx,
            request=query,
            model=chat_model,
            response_duration=res_time,
            token_usage=token_usage,
            steps=[step],
        )

    def get_template(self) -> str:
        return self.template

    def get_config(self) -> RetrievalConfig:
        return self.behaviour


class FormulaKnowledgeAgent(KnowledgeAgent):
    """
    This agent retrieves all the formula knowledge for the user query.
    Based on these documents, the agent generates a response with the needed formula knowledge and an explanation, how to use it.
    """

    name: str = "formula_knowledge_agent"

    TEMPLATE: str = """You are an expert in retrieving and presenting DevExpress formula knowledge. Based on the user query and the retrieval configuration, generate a response with the needed formula knowledge.
  
    Query: {query}
    
    Use the following context to generate the response:
    {context}
    
    Your response should include:
    1. Relevant functions and methods for formula creation.
    2. An explanation of how to use each function or method in the context of the query."""
    # Ensure the information is accurate and concise.

    RETRIEVAL_BEHAVIOUR: RetrievalConfig = RetrievalConfig(
        retriever=RetrieverConfig(
            retriever_name="formula_retriever",
            retriever_type=RetrieverType.VECTOR,
            index_name="mergedfunctionindex",
            embedding_model=AvailableModels.EMBEDDING_2,
            retriever_select=["chunk", "metadata"],
            field_mappings={
                "name": "metadata.function_name",
                "summary": "chunk",
                "content": "chunk",
            },
        ),
        context_key="context",
        pre_retrieval_type=PreRetrievalType.DEFAULT,
        post_retrieval_type=PostRetrievalType.DEFAULT,
    )

    def __init__(self):
        super().__init__(self.name, self.TEMPLATE, self.RETRIEVAL_BEHAVIOUR)


class BusinessLogicKnowledgeAgent(KnowledgeAgent):
    """
    This agent retrieves all the business logic knowledge for the user query.
    Based on these documents, the agent generates a response with the needed business logic knowledge and an explanation, how to use it.
    """
    name: str = "business_logic_knowledge_agent"

    TEMPLATE: str = """You are an expert in retrieving and presenting business logic knowledge. Based on the user query and the retrieval configuration, generate a response with the needed business logic knowledge.
  
    Query: {query}
    
    Use the following context to generate the response:
    {context}
    
    Your response should include:
    1. Relevant properties and domain-specific details needed for the formula.
    2. An explanation of how each property or detail is relevant to the query."""
    # Ensure the information is accurate and concise.

    RETRIEVAL_BEHAVIOUR: RetrievalConfig = RetrievalConfig(
        retriever=RetrieverConfig(
            retriever_name="business_logic_retriever",
            retriever_type=RetrieverType.VECTOR,
            index_name="domain_knowledge",
            embedding_model=AvailableModels.EMBEDDING_3_LARGE,
            retriever_select=["name", "summary", "content"],
            field_mappings={
                "name": "name",
                "summary": "summary",
                "content": "content",
            },
        ),
        context_key="context",
        pre_retrieval_type=PreRetrievalType.DEFAULT,
        post_retrieval_type=PostRetrievalType.DEFAULT,
    )

    def __init__(self):
        super().__init__(self.name, self.TEMPLATE, self.RETRIEVAL_BEHAVIOUR)
