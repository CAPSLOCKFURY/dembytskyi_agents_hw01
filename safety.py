"""
Завдання 3 (частина 1). Захисні механізми агента:
- max_steps (обмеження кількості ітерацій циклу)
- timeout (загальний тайм-аут виконання через time.monotonic)
- LoopDetector (детекція повторюваних tool calls поспіль)
"""

import hashlib
import json
import time


DEFAULT_MAX_STEPS = 10
DEFAULT_TIMEOUT_SECONDS = 90.0
DEFAULT_MAX_REPEATS = 3


class LoopDetector:
    """Детекція зациклення агента — виявлення повторюваних викликів інструментів."""

    def __init__(self, max_repeats: int = DEFAULT_MAX_REPEATS):
        self.recent_calls: list[str] = []
        self.max_repeats = max_repeats

    def reset(self) -> None:
        """Очистити історію викликів перед новим запуском."""
        self.recent_calls.clear()

    def check(self, tool_name: str, args: dict) -> bool:
        """Перевірити, чи виклик інструменту з тими самими аргументами повторюється N разів поспіль.

        Args:
            tool_name: Назва інструменту.
            args: Словник аргументів виклику.

        Returns:
            True, якщо зациклення виявлено (останні max_repeats викликів ідентичні).
        """
        normalized_args = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
        call_signature = f"{tool_name}:{normalized_args}"
        call_hash = hashlib.md5(call_signature.encode("utf-8")).hexdigest()
        self.recent_calls.append(call_hash)

        if len(self.recent_calls) >= self.max_repeats:
            last_n = self.recent_calls[-self.max_repeats :]
            if len(set(last_n)) == 1:
                return True
        return False


class ExecutionGuard:
    """Контролер обмежень часу виконання (timeout) та кількості кроків (max_steps)."""

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.start_time = time.monotonic()

    def reset(self) -> None:
        """Скинути таймер перед новим запуском."""
        self.start_time = time.monotonic()

    def elapsed_seconds(self) -> float:
        """Час, що минув з початку запуску (у секундах)."""
        return time.monotonic() - self.start_time

    def is_timeout(self) -> bool:
        """Перевірити, чи перевищено ліміт часу виконання."""
        return self.elapsed_seconds() >= self.timeout_seconds

    def is_max_steps_reached(self, current_step: int) -> bool:
        """Перевірити, чи досягнуто максимальної кількості кроків."""
        return current_step >= self.max_steps
