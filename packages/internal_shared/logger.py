import os
from functools import lru_cache
from logging import Logger, getLogger


class CustomLogger(Logger):
    def __init__(self, name: str):
        super().__init__(name)

    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra=None,
        stack_info=False,
        stacklevel=1,
    ):
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)


@lru_cache(maxsize=None)
def get_logger(name: str) -> Logger:
    if os.getenv("FASTAPI_ENV"):
        return getLogger("uvicorn")
    try:
        import uvicorn

        return getLogger("uvicorn")
    except ImportError:
        return CustomLogger(name)
