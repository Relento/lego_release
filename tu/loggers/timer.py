import time
from contextlib import contextmanager


@contextmanager
def print_time(name) -> float:
    t = time.perf_counter()
    yield
    print("[TIME]", name, time.perf_counter() - t)


@contextmanager
def named_timeit(name, store_dict) -> float:
    if name not in store_dict:
        store_dict[name] = 0
    t = time.perf_counter()
    yield
    store_dict[name] += time.perf_counter() - t
