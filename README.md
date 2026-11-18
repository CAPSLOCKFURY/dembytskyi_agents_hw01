# ReAct-агент для фінансового та інвестиційного аналізу

## 1. Опис проєкту
ReAct-агент (Reasoning + Acting) у **LangGraph** для вирішення фінансових та інвестиційних задач на базі **Google Gemini API** (`ChatGoogleGenerativeAI`). Агент використовує 4 інструменти з валідацією **Pydantic v2**, підтримує structured outputs (`AgentStructuredResponse`), захисні механізми (`max_steps`, `timeout`, `LoopDetector`) та логування траєкторії у JSON.

## 2. Доменна задача
- **Домен:** Персональні фінанси та інвестиції.
- **Що вирішує агент:** Конвертація валют, розрахунок дохідності депозитів/інвестицій, аналіз показників акцій/ETF та розрахунок кредитних платежів.
- **Інструменти (tools):**
  - `convert_currency` — конвертація між `USD`, `EUR`, `UAH`, `GBP`, `PLN`.
  - `calculate_compound_interest` — розрахунок складного відсотка з щомісячним поповненням.
  - `get_stock_info` — ціна, P/E, дивіденди, сектор і ризик для тікерів `AAPL`, `MSFT`, `NVDA`, `KO`, `VOO`.
  - `calculate_loan_payment` — ануїтетний щомісячний платіж і переплата за кредитом.

## 3. Архітектура
- **LLM-провайдер:** Google Gemini (`gemini-3.1-flash-lite` / `gemini-3-flash-preview` із fallback на резервні моделі Gemini 3 Flash при `503`/`429`).
- **Граф (`StateGraph`):** `START → agent → (should_continue) → tools → agent → formatter → END`. Вузол `formatter` повертає структуровану відповідь (`answer`, `confidence`, `sources`).
- **Захисні механізми:** `max_steps = 10`, `timeout = 90.0` сек, `LoopDetector(max_repeats = 3)`.

## 4. Інструкція запуску
1. Перейти в директорію проєкту: `cd agents_hw01`
2. Встановити залежності: `pip install -r requirements.txt`
3. Додати API-ключ у файл `.env`: `GOOGLE_API_KEY="ваш_ключ"`
4. Запустити тести: `python test_runner.py`

## 5. Результати тестування
Підсумкова таблиця за `test_results.json`:

| Test ID | Складність | Кроків | Tool calls | Час, мс | Статус |
|---------|------------|--------|------------|---------|--------|
| TC-001 | simple | 2 | `convert_currency` × 1 | 44827 | ✅ success |
| TC-002 | simple | 2 | `get_stock_info` × 1 | 43672 | ✅ success |
| TC-003 | medium | 2 | `get_stock_info` × 2 | 3922 | ✅ success |
| TC-004 | medium | 3 | `convert_currency` × 1, `calculate_compound_interest` × 1 | 34702 | ✅ success |
| TC-005 | complex | 3 | `calculate_loan_payment` × 1, `get_stock_info` × 2, `convert_currency` × 1 | 10907 | ✅ success |

## 6. Аналіз результатів
- **Чи правильно агент обирав tools?** Так, у всіх 5 тест-кейсах агент обрав потрібні інструменти з валідними аргументами: паралельно викликав `get_stock_info` для порівняння активів (`TC-003`, `TC-005`) і послідовно передавав результат одного інструменту в інший (`TC-004`, `TC-005`).
- **Де агент помилявся і чому?** Помилок вибору інструментів чи галюцинацій параметрів не було. Виникали лише мережеві помилки `503`/`429` на Free Tier Gemini API, які вирішено автоматичним fallback-перемиканням між моделями Gemini.
- **Кроки: simple vs complex.** Прості та паралельні запити (`TC-001`..`TC-003`) виконуються за 2 кроки, а послідовно залежні (`TC-004`, `TC-005`) — за 3 кроки. Кількість кроків залежить від глибини залежностей даних, а не від загальної кількості tool calls.
- **Чи досягав агент max_steps або timeout?** На основних тестах — ні (максимум 3 кроки з 10, до 44.8 с із 90 с). Спрацювання `max_steps`, `timeout` і `LoopDetector` окремо протестовано у `demo_safety_mechanisms()`.
- **Залежність часу виконання від складності.** Локальні інструменти працюють за `0–16 мс`; ~99% часу займають мережеві звернення до LLM API та можливі retry при перевантаженні серверів.
- **Випадки, коли агент НЕ використав tool, хоч мав би.** Таких випадків не зафіксовано завдяки чітким docstrings, `Field(description=...)` та системному промпту.

## 7. Висновки та обмеження
- **Що працює добре:** Pydantic v2 валідація, комбінація паралельних і послідовних tool calls, structured output та логування в `trajectory.json`.
- **Що можна покращити:** Замінити mock-дані на реальні фінансові API (НБУ, Yahoo Finance) і додати кешування.
- **Відомі обмеження:** Квоти безкоштовного тарифу Gemini API та фіксований перелік валют і тікерів у локальній базі.

## 8. Структура файлів
- `tools.py` — Pydantic-схеми, 4 інструменти та unit-тести
- `agent.py` — LangGraph граф та демонстрація захисних механізмів
- `safety.py` — `ExecutionGuard` (`max_steps`, `timeout`) та `LoopDetector`
- `logger.py` — `TrajectoryLogger`
- `test_runner.py` — запуск тест-кейсів
- `trajectory.json` — лог траєкторії
- `test_results.json` — результати тестів
- `requirements.txt` — залежності
