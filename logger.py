"""
Завдання 3 (частина 2). Логування траєкторії виконання агента у JSON-файл (trajectory.json).
"""

import json
import time
from datetime import datetime, timezone
from typing import Any


class TrajectoryLogger:
    """Структуроване логування траєкторії виконання ReAct-агента у JSON."""

    def __init__(self):
        self.steps: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.start_time = time.monotonic()
        self.current_run_id: str | None = None

    def start_run(self, run_id: str, query: str) -> None:
        """Почати новий сеанс запису для конкретного запиту."""
        self.current_run_id = run_id
        self.start_time = time.monotonic()

    def log_step(
        self,
        step_num: int,
        node: str,
        input_data: str,
        output_data: str,
        tool_calls: list[Any] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        """Записати один крок виконання вузла графа."""
        now_elapsed = int((time.monotonic() - self.start_time) * 1000)
        entry = {
            "run_id": self.current_run_id or "default",
            "step_number": step_num,
            "node_name": node,
            "input": str(input_data)[:500],
            "output": str(output_data)[:500],
            "tool_calls": tool_calls or [],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "duration_ms": duration_ms if duration_ms is not None else now_elapsed,
            "elapsed_ms": now_elapsed,
        }
        self.steps.append(entry)

    def save(self, filepath: str = "trajectory.json") -> None:
        """Зберегти повну траєкторію виконання у JSON-файл."""
        total_time_ms = int((time.monotonic() - self.start_time) * 1000)
        payload = {
            "total_steps": len(self.steps),
            "total_time_ms": total_time_ms,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "trajectory": self.steps,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Траєкторію збережено у '{filepath}' (всього кроків: {len(self.steps)})")
