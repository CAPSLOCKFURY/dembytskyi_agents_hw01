"""
Завдання 4. Запуск 5 тест-кейсів різної складності на ReAct-агенті у LangGraph,
збір результатів у test_results.json та збереження повної траєкторії у trajectory.json.
"""

import json
import time
from langchain_core.messages import HumanMessage

from agent import _extract_text, build_react_agent, demo_safety_mechanisms
from logger import TrajectoryLogger
from tools import run_tool_unit_tests


# ── Тест-кейси різної складності (Simple, Medium, Complex) ──────────────
TEST_CASES = [
    {
        "id": "TC-001",
        "query": "Скільки буде 25000 UAH у доларах США (USD)?",
        "expected": "Конвертація 25000 UAH у USD за курсом 0.024 (600.00 USD)",
        "complexity": "simple",
    },
    {
        "id": "TC-002",
        "query": "Яка поточна ціна та фінансові показники акцій Apple (AAPL)?",
        "expected": "Інформація про AAPL: ціна $228.50, P/E 34.2, дивіденди 0.44%, рівень ризику medium",
        "complexity": "simple",
    },
    {
        "id": "TC-003",
        "query": "Порівняй акції Microsoft (MSFT) та NVIDIA (NVDA): яка з них дешевша за мультиплікатором P/E і має менший ризик?",
        "expected": "Порівняння MSFT (P/E 36.8, ризик low) та NVDA (P/E 52.4, ризик high), висновок на користь MSFT за нижчим P/E та ризиком",
        "complexity": "medium",
    },
    {
        "id": "TC-004",
        "query": "Я маю 100000 UAH. Переведи їх у USD і порахуй, скільки я накопичу за 5 років, якщо покладу цю суму в доларах під 8% річних із щомісячним поповненням 100 USD.",
        "expected": "Конвертація 100000 UAH -> 2400 USD та розрахунок складного відсотка на 5 років під 8% річних із поповненням $100/міс (~10921.52 USD)",
        "complexity": "medium",
    },
    {
        "id": "TC-005",
        "query": (
            "Проведи комплексний фінансовий аналіз: я беру кредит 200000 UAH на 24 місяці під 16% річних. "
            "Розрахуй мій щомісячний платіж і переплату в UAH, переведи суму переплати в USD, "
            "а також перевір показники фонду VOO та акцій KO і порадь, який із цих двох активів має вищу дивідендну дохідність."
        ),
        "expected": (
            "Розрахунок кредиту 200000 UAH (платіж ~9792.62 UAH, переплата ~35022.92 UAH), "
            "конвертація переплати в USD (~840.55 USD), порівняння дивідендів VOO (1.32%) та KO (3.08%)"
        ),
        "complexity": "complex",
    },
]


def run_all_tests() -> None:
    """Виконати unit-тести інструментів, демонстрацію захисних механізмів та 5 тест-кейсів агента."""
    # 1. Unit-тести інструментів (Завдання 1)
    run_tool_unit_tests()

    # 2. Демонстрація захисних механізмів (Завдання 3)
    demo_safety_mechanisms()

    # 3. Основне тестування ReAct-агента з Gemini (Завдання 2, 3, 4)
    print("=== [Завдання 2, 3, 4] Запуск ReAct-агента на тест-кейсах ===")
    trajectory_logger = TrajectoryLogger()
    app = build_react_agent(
        max_steps=10,
        timeout_seconds=90.0,
        max_repeats=3,
        logger=trajectory_logger,
        use_structured_formatter=True,
    )

    results = []
    for tc in TEST_CASES:
        print(f"\n▶ Running {tc['id']} [{tc['complexity']}]: {tc['query'][:75]}...")
        trajectory_logger.start_run(tc["id"], tc["query"])
        start = time.monotonic()

        try:
            state_out = app.invoke(
                {
                    "messages": [HumanMessage(content=tc["query"])],
                    "step_count": 0,
                    "structured_response": None,
                    "stop_reason": None,
                }
            )
            raw_last = _extract_text(state_out["messages"][-1].content)
            structured = state_out.get("structured_response") or {}
            actual_answer = structured.get("answer") or raw_last
            steps = state_out.get("step_count", 0)
            stop_reason = state_out.get("stop_reason")
            status = "partial" if stop_reason else "success"

            tool_calls = []
            for msg in state_out.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc_call in msg.tool_calls:
                        tool_calls.append(tc_call.get("name", "unknown"))
        except Exception as e:
            actual_answer = f"Error: {e}"
            structured = {"answer": actual_answer, "confidence": 0.0, "sources": []}
            steps = -1
            status = "error"
            stop_reason = "exception"
            tool_calls = []

        elapsed_ms = int((time.monotonic() - start) * 1000)

        results.append(
            {
                "test_id": tc["id"],
                "query": tc["query"],
                "expected": tc["expected"],
                "actual": actual_answer[:500],
                "structured_output": structured,
                "status": status,
                "stop_reason": stop_reason,
                "steps": steps,
                "tool_calls": tool_calls,
                "elapsed_ms": elapsed_ms,
                "complexity": tc["complexity"],
            }
        )
        print(
            f"  ✔ {tc['id']}: status={status}, steps={steps}, "
            f"tools={tool_calls}, time={elapsed_ms}ms"
        )
        print(f"  Structured response: confidence={structured.get('confidence')}, sources={structured.get('sources')}")

    # Збереження траєкторії (trajectory.json)
    trajectory_logger.save("trajectory.json")

    # Збереження результатів тестування (test_results.json)
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nРезультати збережено у 'test_results.json' ({len(results)} тест-кейсів):")
    for r in results:
        print(
            f"  {r['test_id']} ({r['complexity']}): {r['status']} | "
            f"{r['steps']} steps | tools={r['tool_calls']} | {r['elapsed_ms']}ms"
        )


if __name__ == "__main__":
    run_all_tests()
