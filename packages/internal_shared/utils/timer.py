from functools import wraps
from logging import Logger, getLogger
import time
from typing import Any, Callable

logger: Logger = getLogger("uvicorn")


def timer(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(
            f"PERF: '{func.__qualname__:<50}' executed in {elapsed_time_ms:>15.5f} ms"
        )
        return result

    return wrapper


def async_timer(func: Callable) -> Callable:
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time_ms = (end_time - start_time) * 1000
        logger.info(
            f"PERF: '{func.__qualname__:<50}' executed in {elapsed_time_ms:>15.5f} ms"
        )
        return result

    return wrapper
