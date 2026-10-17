#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Jakub Barylski"
__maintainer__ = "Jakub Barylski"
__license__ = "GNU GENERAL PUBLIC LICENSE"
__email__ = "jakub.barylski@gmail.com"

import functools
# python
import os

# limit BLAS/MKL/OpenBLAS threads before any numpy import
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import inspect
import pickle
import sys
import warnings
from functools import wraps
from multiprocessing import cpu_count
from pathlib import Path
import subprocess
import signal
from typing import Callable, Dict, List, Hashable, Any, Generator, Optional

from loguru import logger
from mpire import WorkerPool


# LOGGING format and configuration
def loguru_showwarning(message, category, filename, lineno, file=None, line=None):
    filename = Path(filename).name
    logger.warning(f"{filename} {category.__name__}: {message}")


logger.remove()
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>: <level>{message}</level>"
logger.add(sys.stderr, format=log_format)
warnings.showwarning = loguru_showwarning

# DEFAULTS
default_threads = max(cpu_count() - 1, 1)
seed_max = 2 ** 32 - 1


def frantic_search(dictionary: Dict[Hashable, Any],
                   *possible_keys: Hashable):
    """
    Find first of the several keys that are in a dictionary and return the value
    :param dictionary: dictionary to search
    :param possible_keys: any number of possible keys (preferred first)
    :return: dictionary[first_found_key]
    """
    for key in possible_keys:
        if key in dictionary:
            return dictionary[key]
    missed_keys = ', '.join([str(k) for k in possible_keys])
    raise KeyError(f'Found none of the: {missed_keys}')


# PARALLELIZATION


class Parallel(WorkerPool):
    """
    Inherits from WorkerPool. Processes tasks by passing the primary
    argument positionally and constants as keywords.
    """

    def __init__(self,
                 function,
                 input_collection,
                 kwargs: Dict[str, Any] = None,
                 n_jobs: int = default_threads,
                 bar: bool = True,
                 description: str = None,
                 chunk_size: int = 100):
        # Initialize the parent WorkerPool
        super().__init__(n_jobs=n_jobs)

        kwargs = kwargs or {}

        def dispatch(*pos_args):
            return function(*pos_args, **kwargs)

        self.results = self.map_unordered(func=dispatch,
                                          iterable_of_args=self.argument_feed(input_collection),
                                          progress_bar=bar,
                                          progress_bar_options={"desc": description} if description else None,
                                          chunk_size=chunk_size)

    @staticmethod
    def argument_feed(input_collection) -> Generator[tuple, None, None]:
        for item in input_collection:
            if isinstance(item, tuple):
                positional_args = item
            else:
                positional_args = (item,)
            yield positional_args



def shell(command: List[str],
          stdout: str = 'suppress',
          stdin: Optional[bytes] = None,
          timeout: Optional[float] = None) -> Optional[bytes]:
    """
    Run external CLI commands, ensuring all sub-processes in the process
    group are fully terminated on timeout or failure.
    """
    sanitized_command = [str(c) for c in command]
    logger.info(" ".join(sanitized_command))

    # Configure standard streams
    if stdout == 'suppress':
        out_stream, err_stream = subprocess.DEVNULL, subprocess.DEVNULL
    elif stdout == 'capture':
        out_stream, err_stream = subprocess.PIPE, subprocess.PIPE
    else:
        out_stream, err_stream = None, None

    # start_new_session=True sets PGID = process.pid
    process = subprocess.Popen(
        sanitized_command,
        stdout=out_stream,
        stderr=err_stream,
        stdin=subprocess.PIPE if stdin is not None else None,
        start_new_session=True
    )

    try:
        out, err = process.communicate(input=stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        # Send SIGKILL to the entire process group (riboseek + worker progeny)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # Process already exited before signal delivery

        process.wait()  # Reap process to prevent zombie status in Linux kernel
        raise TimeoutError(f'Command "{" ".join(sanitized_command)}" timed out after {timeout} seconds.')

    if process.returncode != 0:
        err_msg = err.decode('utf-8', errors='replace') if err else "Process failed silently"
        raise ChildProcessError(
            f'Command "{" ".join(sanitized_command)}" exited with code {process.returncode}:\n{err_msg}')

    return out if stdout == 'capture' else None


def total_time(func):
    """
    Simple timeit decorator
    :param func: function to wrap
    :return:
    """

    @wraps(func)
    def timed(*args, **kwargs):
        from time import time
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        total_time = end_time - start_time
        human_readable_time = f'{int(total_time // 3600):02d}:' \
                              f'{int((total_time % 3600) // 60):02d}:' \
                              f'{int(total_time % 60):02d}'
        logger.info(f'Function "{func.__name__}" executed in {human_readable_time} (hh:mm:ss)')
        return result

    return timed


def checkpoint(funct: Callable):
    """
    Simple serialization decorator
    that saves the function result
    if exacted output file doesn't exist or is empty
    or read it if it is non-empty
    @param funct: function to wrap
    @param pickle_path: a path to an output file
    @param serialization_method: a module used for serialization (either joblib or pickle)
    @return:
    """

    signature = inspect.signature(funct)

    @wraps(funct)
    def save_checkpoint(*args, **kwargs):

        bound_args = signature.bind(*args, **kwargs)
        pickle_path = Path(bound_args.arguments.get('pickle_path',
                                                    signature.parameters['pickle_path'].default))
        if pickle_path:
            try:
                with pickle_path.open('rb') as file_object:
                    result = pickle.load(file_object)
                logger.info(f'\ntemporary file read from: {pickle_path.as_posix()}\n', flush=True)
                return result
            except (FileNotFoundError, IOError, EOFError):
                sys.setrecursionlimit(5000)
                result = funct(*args, **kwargs)
                with pickle_path.open('wb') as out:
                    pickle.dump(result, out)
                logger.info(f'\ntemporary file stored at: {pickle_path.as_posix()}\n', flush=True)
                return result

    return save_checkpoint



def timeout_signal(seconds=1800):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            def _handle_timeout(signum, frame):
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds.")

            # Set signal handler and alarm
            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                # Disable alarm and restore previous handler
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator