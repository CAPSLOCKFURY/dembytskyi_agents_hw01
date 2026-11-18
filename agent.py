"""
Завдання 2 та Завдання 3. Побудова LangGraph ReAct-агента з циклом LLM–tools–LLM,
structured outputs та захисними механізмами (max_steps, timeout, LoopDetector, TrajectoryLogger).
"""

import operator
import os
import time
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from logger import TrajectoryLogger
from safety import (
    DEFAULT_MAX_REPEATS,
    DEFAULT_MAX_STEPS,
    DEFAULT_TIMEOUT_SECONDS,
    ExecutionGuard,
    LoopDetector,
)
from tools import ALL_TOOLS

# Завантажуємо змінні середовища з файлу .env
load_dotenv()


# ── Pydantic-схема для Structured Output фінальної відповіді ─────────────
class AgentStructuredResponse(BaseModel):
    """Структурована фінальна відповідь фінансового ReAct-агента."""

    answer: str = Field(
        description="Детальна, чітка та обґрунтована відповідь українською мовою з точними цифрами"
    )
    confidence: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Рівень впевненості у відповіді (від 0.0 до 1.0)",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Список використаних інструментів або джерел даних (наприклад, ['convert_currency', 'get_stock_info'])",
    )


# ── Стан графа (AgentState) ──────────────────────────────────────────────
class AgentState(TypedDict):
    """Стан ReAct-агента у LangGraph."""

    messages: Annotated[list[BaseMessage], operator.add]
    step_count: int
    structured_response: dict[str, Any] | None
    stop_reason: str | None


SYSTEM_PROMPT = (
    "Ти — професійний фінансовий та інвестиційний AI-асистент. "
    "Для розрахунків валют, складних відсотків, кредитних платежів та перевірки цін акцій/ETF "
    "ОБОВ'ЯЗКОВО використовуй надані інструменти (tools), не вигадуй курси чи цифри самостійно. "
    "Відповідай українською мовою, чітко наводячи отримані числові результати."
)


def create_gemini_llm(model_name: str | None = None) -> ChatGoogleGenerativeAI:
    """Ініціалізувати клієнт Google Gemini з API-ключем із .env."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "API-ключ не знайдено! Перевірте наявність GOOGLE_API_KEY у файлі .env"
        )
    # Використовуємо модель Gemini з підтримкою tool calling.
    # Для моделей Gemini 3 температура залишається за замовчуванням (1.0).
    selected_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return ChatGoogleGenerativeAI(
        model=selected_model,
        google_api_key=api_key,
        max_retries=1,
        timeout=20,
    )


def invoke_with_fallback(
    messages: list[BaseMessage],
    tools: list[Any] | None = None,
    structured_schema: type[BaseModel] | None = None,
    primary_model: str = "gemini-3.1-flash-lite",
) -> Any:
    """Виклик Gemini із автоматичним fallback на резервну модель Gemini у разі тимчасового 503/504/429 перевантаження."""
    candidate_models = [
        primary_model,
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
    ]
    # Усуваємо дублікати зі збереженням порядку
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_err: Exception | None = None
    for model_name in models_to_try:
        try:
            llm = create_gemini_llm(model_name)
            if structured_schema is not None:
                runnable = llm.with_structured_output(structured_schema)
            elif tools is not None:
                runnable = llm.bind_tools(tools)
            else:
                runnable = llm
            return runnable.invoke(messages)
        except Exception as err:
            err_text = str(err)
            last_err = err
            if any(code in err_text for code in ("503", "504", "UNAVAILABLE", "DEADLINE_EXCEEDED", "429", "404")):
                continue
            raise
    raise RuntimeError(f"Усі моделі Gemini тимчасово недоступні: {last_err}")


def _extract_text(content: Any) -> str:
    """Нормалізувати контент повідомлення LLM у текстовий рядок."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def build_react_agent(
    tools: list[Any] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_repeats: int = DEFAULT_MAX_REPEATS,
    logger: TrajectoryLogger | None = None,
    custom_llm_callable: Any | None = None,
    use_structured_formatter: bool = True,
):
    """Побудувати та скомпілювати LangGraph ReAct-агента із захисними механізмами та логуванням.

    Args:
        tools: Список інструментів (за замовчуванням ALL_TOOLS).
        max_steps: Максимальна кількість кроків циклу.
        timeout_seconds: Максимальний час виконання у секундах.
        max_repeats: Поріг детекції зациклення однакових tool calls.
        logger: Екземпляр TrajectoryLogger для запису траєкторії.
        custom_llm_callable: Опціональна функція замість реального LLM (для детермінованих демо захисних механізмів).
        use_structured_formatter: Чи формувати structured output (AgentStructuredResponse) у фіналі.
    """
    active_tools = tools if tools is not None else ALL_TOOLS
    tool_node = ToolNode(active_tools)
    guard = ExecutionGuard(max_steps=max_steps, timeout_seconds=timeout_seconds)
    loop_detector = LoopDetector(max_repeats=max_repeats)
    traj_logger = logger if logger is not None else TrajectoryLogger()

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Вузол agent: перевіряє захисні ліміти, викликає LLM та перевіряє LoopDetector."""
        node_start = time.monotonic()
        step = state.get("step_count", 0)
        messages = list(state.get("messages", []))

        if step == 0:
            guard.reset()
            loop_detector.reset()

        last_input_text = _extract_text(messages[-1].content) if messages else ""

        # 1. Перевірка Timeout
        if guard.is_timeout():
            elapsed_s = guard.elapsed_seconds()
            msg = AIMessage(
                content=(
                    f"[TIMEOUT] Перевищено ліміт часу виконання ({timeout_seconds} сек, "
                    f"минуло {elapsed_s:.2f} сек). Повертаю часткову відповідь після {step} кроків."
                )
            )
            duration = int((time.monotonic() - node_start) * 1000)
            traj_logger.log_step(
                step_num=step + 1,
                node="agent",
                input_data=last_input_text,
                output_data=msg.content,
                tool_calls=[],
                duration_ms=duration,
            )
            return {
                "messages": [msg],
                "step_count": step,
                "stop_reason": "timeout",
            }

        # 2. Перевірка Max Steps
        if guard.is_max_steps_reached(step):
            msg = AIMessage(
                content=(
                    f"[MAX_STEPS] Досягнуто ліміт кроків ({max_steps}). "
                    f"Часткова відповідь: виконано {step} кроків."
                )
            )
            duration = int((time.monotonic() - node_start) * 1000)
            traj_logger.log_step(
                step_num=step + 1,
                node="agent",
                input_data=last_input_text,
                output_data=msg.content,
                tool_calls=[],
                duration_ms=duration,
            )
            return {
                "messages": [msg],
                "step_count": step,
                "stop_reason": "max_steps",
            }

        # Додаємо системний промпт, якщо його ще немає
        llm_messages = messages
        if not messages or not isinstance(messages[0], SystemMessage):
            llm_messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Виклик LLM (або тестової функції)
        if custom_llm_callable is not None:
            response = custom_llm_callable(llm_messages, step)
        else:
            response = invoke_with_fallback(llm_messages, tools=active_tools)

        # 3. Перевірка LoopDetector на повторювані tool calls
        tool_calls_list = getattr(response, "tool_calls", None) or []
        recorded_calls = []
        for tc in tool_calls_list:
            t_name = tc.get("name", "unknown")
            t_args = tc.get("args", {})
            recorded_calls.append({"name": t_name, "args": t_args})
            if loop_detector.check(t_name, t_args):
                loop_msg = AIMessage(
                    content=(
                        f"[LOOP_DETECTED] Виявлено зациклення: інструмент '{t_name}' "
                        f"з аргументами {t_args} викликано {max_repeats} разів поспіль. Виконання зупинено."
                    )
                )
                duration = int((time.monotonic() - node_start) * 1000)
                traj_logger.log_step(
                    step_num=step + 1,
                    node="agent",
                    input_data=last_input_text,
                    output_data=loop_msg.content,
                    tool_calls=recorded_calls,
                    duration_ms=duration,
                )
                return {
                    "messages": [loop_msg],
                    "step_count": step + 1,
                    "stop_reason": "loop_detected",
                }

        duration = int((time.monotonic() - node_start) * 1000)
        out_text = _extract_text(response.content)
        if not out_text and recorded_calls:
            out_text = f"Tool calls: {[c['name'] for c in recorded_calls]}"

        traj_logger.log_step(
            step_num=step + 1,
            node="agent",
            input_data=last_input_text,
            output_data=out_text,
            tool_calls=recorded_calls,
            duration_ms=duration,
        )

        return {
            "messages": [response],
            "step_count": step + 1,
            "stop_reason": state.get("stop_reason"),
        }

    def tools_node_logged(state: AgentState) -> dict[str, Any]:
        """Вузол tools: виконує tool calls через ToolNode з перехопленням помилок валідації та логуванням."""
        node_start = time.monotonic()
        step = state.get("step_count", 0)
        last_msg = state["messages"][-1]
        t_calls = getattr(last_msg, "tool_calls", []) or []

        # Перевірка timeout перед виконанням інструментів
        if guard.is_timeout():
            elapsed_s = guard.elapsed_seconds()
            tool_messages = [
                ToolMessage(
                    content=f"[TIMEOUT] Виконання інструменту скасовано через перевищення ліміту часу ({elapsed_s:.2f}s).",
                    tool_call_id=tc.get("id", "timeout_id"),
                    name=tc.get("name", "unknown"),
                )
                for tc in t_calls
            ]
            return {"messages": tool_messages}

        try:
            result = tool_node.invoke(state)
            new_messages = result.get("messages", [])
        except Exception as e:
            # Якщо виникла помилка Pydantic-валідації, повертаємо її як ToolMessage, щоб агент міг її обробити
            new_messages = [
                ToolMessage(
                    content=f"Validation/Execution Error: {e}",
                    tool_call_id=tc.get("id", "err_id"),
                    name=tc.get("name", "unknown"),
                )
                for tc in t_calls
            ]

        duration = int((time.monotonic() - node_start) * 1000)
        outputs_combined = " | ".join(_extract_text(m.content) for m in new_messages)
        inputs_combined = ", ".join(
            f"{tc.get('name')}({tc.get('args')})" for tc in t_calls
        )

        traj_logger.log_step(
            step_num=step,
            node="tools",
            input_data=inputs_combined,
            output_data=outputs_combined,
            tool_calls=[tc.get("name", "unknown") for tc in t_calls],
            duration_ms=duration,
        )
        return {"messages": new_messages}

    def formatter_node(state: AgentState) -> dict[str, Any]:
        """Вузол formatter: формує гарантований Structured Output (AgentStructuredResponse)."""
        messages = state.get("messages", [])
        final_text = _extract_text(messages[-1].content) if messages else ""
        stop_reason = state.get("stop_reason")

        # Збираємо всі фактично викликані інструменти як sources
        used_tools: list[str] = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name")
                    if name and name not in used_tools:
                        used_tools.append(name)

        # Якщо спрацював захисний механізм або кастомний LLM, формуємо Pydantic-відповідь напряму
        if stop_reason or custom_llm_callable is not None or not use_structured_formatter:
            confidence = 0.3 if stop_reason else 0.95
            structured = AgentStructuredResponse(
                answer=final_text,
                confidence=confidence,
                sources=used_tools,
            )
            return {"structured_response": structured.model_dump()}

        # Використовуємо with_structured_output для структуризації фінальної відповіді
        try:
            prompt = [
                SystemMessage(
                    content=(
                        "Сформуй фінальну структуровану відповідь на основі діалогу. "
                        "Збережи всі числові розрахунки у полі answer, оціни confidence (0..1) "
                        "та перелічи використані інструменти у полі sources."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Текст відповіді агента:\n{final_text}\n\n"
                        f"Використані інструменти: {used_tools}"
                    )
                ),
            ]
            structured_obj = invoke_with_fallback(
                prompt, structured_schema=AgentStructuredResponse
            )
            if isinstance(structured_obj, AgentStructuredResponse):
                if not structured_obj.sources and used_tools:
                    structured_obj.sources = used_tools
                return {"structured_response": structured_obj.model_dump()}
        except Exception:
            pass

        fallback_structured = AgentStructuredResponse(
            answer=final_text,
            confidence=0.9,
            sources=used_tools,
        )
        return {"structured_response": fallback_structured.model_dump()}

    def should_continue(state: AgentState) -> Literal["tools", "formatter"]:
        """Router (conditional edge): якщо LLM повернув tool_calls і немає stop_reason → tools, інакше → formatter."""
        if state.get("stop_reason"):
            return "formatter"
        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "formatter"

    # Побудова графа StateGraph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node_logged)
    graph.add_node("formatter", formatter_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "agent")
    graph.add_edge("formatter", END)

    return graph.compile()


def demo_safety_mechanisms() -> None:
    """Демонстрація роботи захисних механізмів (Завдання 3): max_steps, timeout, LoopDetector."""
    print("=== [Завдання 3] Демонстрація захисних механізмів ===")

    # 1. Демонстрація max_steps
    counter = {"n": 0}

    def varied_tool_llm(messages: list[BaseMessage], step: int) -> AIMessage:
        counter["n"] += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "convert_currency",
                    "args": {
                        "amount": float(100 + counter["n"]),
                        "from_currency": "USD",
                        "to_currency": "UAH",
                    },
                    "id": f"call_step_{counter['n']}",
                }
            ],
        )

    app_max_steps = build_react_agent(
        max_steps=3,
        timeout_seconds=30.0,
        max_repeats=5,
        custom_llm_callable=varied_tool_llm,
    )
    res_steps = app_max_steps.invoke(
        {"messages": [HumanMessage(content="Тест max_steps")], "step_count": 0}
    )
    print(f"1) Max Steps Demo -> stop_reason={res_steps.get('stop_reason')}: {res_steps['messages'][-1].content}")
    assert res_steps.get("stop_reason") == "max_steps"

    # 2. Демонстрація LoopDetector (3 однакових виклики поспіль)
    def repeating_tool_llm(messages: list[BaseMessage], step: int) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_stock_info",
                    "args": {"ticker": "AAPL"},
                    "id": f"call_loop_{step}",
                }
            ],
        )

    app_loop = build_react_agent(
        max_steps=10,
        timeout_seconds=30.0,
        max_repeats=3,
        custom_llm_callable=repeating_tool_llm,
    )
    res_loop = app_loop.invoke(
        {"messages": [HumanMessage(content="Тест LoopDetector")], "step_count": 0}
    )
    print(f"2) LoopDetector Demo -> stop_reason={res_loop.get('stop_reason')}: {res_loop['messages'][-1].content}")
    assert res_loop.get("stop_reason") == "loop_detected"

    # 3. Демонстрація Timeout
    def slow_llm(messages: list[BaseMessage], step: int) -> AIMessage:
        time.sleep(0.15)
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_stock_info",
                    "args": {"ticker": "MSFT" if step % 2 == 0 else "NVDA"},
                    "id": f"call_slow_{step}",
                }
            ],
        )

    app_timeout = build_react_agent(
        max_steps=10,
        timeout_seconds=0.1,
        max_repeats=5,
        custom_llm_callable=slow_llm,
    )
    res_timeout = app_timeout.invoke(
        {"messages": [HumanMessage(content="Тест Timeout")], "step_count": 0}
    )
    print(f"3) Timeout Demo -> stop_reason={res_timeout.get('stop_reason')}: {res_timeout['messages'][-1].content}")
    assert res_timeout.get("stop_reason") == "timeout"
    print("=== Захисні механізми працюють коректно! ===\n")


if __name__ == "__main__":
    demo_safety_mechanisms()
