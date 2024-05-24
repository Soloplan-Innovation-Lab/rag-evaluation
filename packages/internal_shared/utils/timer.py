import time
from typing import Awaitable, Callable, Tuple, TypeVar

R = TypeVar("R")


def time_wrapper(func: Callable[..., R], *args, **kwargs) -> Tuple[float, R]:
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed_time = (end - start) * 1000
    return elapsed_time, result


async def atime_wrapper(
    func: Callable[..., Awaitable[R]], *args, **kwargs
) -> Tuple[float, R]:
    start = time.perf_counter()
    result = await func(*args, **kwargs)
    end = time.perf_counter()
    elapsed_time = (end - start) * 1000
    return elapsed_time, result
