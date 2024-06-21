from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Generator, List, Tuple
from functools import lru_cache


def load_xml_file(file_path: str | Path) -> str:
    with open(file_path, "r") as file:
        return file.read()


def should_exclude_path(file_path: Path) -> bool:
    # Exclude folders containing "SOLOPLAN" (case insensitive)
    if any("SOLOPLAN" in part.upper() for part in file_path.parts):
        return True
    # Exclude "TestSystem" subfolder
    if "TestSystem" in file_path.parts:
        return True
    return False


def should_exclude_file(file_path: Path) -> bool:
    # Exclude files with "Options" or "Property" in their name
    if "Options" in file_path.name or "Property" in file_path.name:
        return True
    return False


def get_customer_workflows(
    filter: Tuple[Callable[[Path], bool], ...] = None
) -> Generator[Path, None, None]:
    for file in Path("/workspace/data/customer_data/workflows").rglob("*.xml"):
        # Apply filters, if any
        if filter:
            # If any filter returns True, continue to the next file
            if any(f(file) for f in filter):
                continue
        yield file


@lru_cache(maxsize=10)
def get_customer_workflows_cached(
    filter: Tuple[Callable[[Path], bool], ...] = None
) -> List[Path]:
    return list(get_customer_workflows(filter))


def get_workflows_content(
    filter: List[Callable[[Path], bool]] = None, use_default_filter: bool = True
):
    if use_default_filter:
        filter = [should_exclude_path, should_exclude_file]
    filter_tuple = tuple(filter) if filter else None
    files = get_customer_workflows_cached(filter_tuple)
    with ThreadPoolExecutor() as executor:
        # Submit all file reading tasks to the executor
        future_to_file = {executor.submit(load_xml_file, file): file for file in files}
        for future in as_completed(future_to_file):
            yield future.result()
