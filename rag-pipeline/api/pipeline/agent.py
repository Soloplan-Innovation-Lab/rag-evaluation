import asyncio
from internal_shared.models.chat import AgentChatRequest, ChatResponse
from agents import FormulaKnowledgeAgent, BusinessLogicKnowledgeAgent, FormulaChatAgent

# currently, this is only executed through a jupyter notebook, rather than mapped to an API endpoint
async def execute_pipeline(request: AgentChatRequest, chat_id: str):
    formula_agent = FormulaKnowledgeAgent()
    business_logic_agent = BusinessLogicKnowledgeAgent()

    f_fn = formula_agent.execute_async(request.query, chat_id)
    b_fn = business_logic_agent.execute_async(request.query, chat_id)

    f_res, b_res = await asyncio.gather(f_fn, b_fn)

    context = f"FormulaAgent response: {f_res.response}\nBusinessLogicAgent response: {b_res.response}"

    chat_agent = FormulaChatAgent()
    response = await chat_agent.execute_async(request.query, context, chat_id)

    return ChatResponse(
        response=response.response,
        documents=f_res.documents + b_res.documents,
        request=request.query,
        model=response.model,
        response_duration=response.response_duration,
        token_usage=response.token_usage,
        steps=f_res.steps + b_res.steps,
        chat_session_id=chat_id,
    )
