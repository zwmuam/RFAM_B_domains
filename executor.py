# fmt: off
"""
System thread detection, parallel process pools, external CLI command execution,
serialization checkpoints, and logging utilities.

Consolidates all multiprocessing and process-execution abstractions into a unified,
fork-safe, and thread-safe implementation.
"""

# built-ins
import inspect
import multiprocessing as mp
import os
import pickle
import signal
import subprocess
import sys
import threading
import warnings
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Collection, Dict, Generator, Hashable, List, Optional, Sequence, Tuple, Union

# standard libraries
import numpy as np

# internal repository/package imports
from loguru import logger
from mpire import WorkerPool

# Limit BLAS/MKL/OpenBLAS threads before any numerical operations
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')


# LOGGING format and configuration
def loguru_showwarning(message: Any,
                       category: Any,
                       filename: str,
                       lineno: int,
                       file: Any = None,
                       line: Any = None) -> None:
    """
    Redirect Python standard warnings to the Loguru logger output.

    :param message: Warning message instance or string.
    :param category: Warning category class.
    :param filename: Source file name generating the warning.
    :param lineno: Line number where warning was triggered.
    :param file: Optional file stream override.
    :param line: Optional line contents override.
    """
    file_name = Path(filename).name
    logger.warning(f"{file_name} {category.__name__}: {message}")


logger.remove()
log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> <cyan>{function}</cyan>: "
              "<level>{message}</level>")
logger.add(sys.stderr, format=log_format)
warnings.showwarning = loguru_showwarning


def get_default_cpu_threads(requested_cpus: Optional[int] = None) -> int:
    """
    Calculate the optimal CPU thread count based on system availability and user requests.

    :param requested_cpus: Optional explicit integer count of requested CPUs.
    :return: Integer count of threads to utilize (defaults to available cores minus 1).
    """
    if requested_cpus is not None and requested_cpus > 0:
        return requested_cpus
    try:
        import psutil
        available_cores: Optional[int] = psutil.cpu_count(logical=True)
    except Exception:
        available_cores = os.cpu_count()

    if not available_cores or available_cores <= 1:
        return 1
    return available_cores - 1


# Default threads alias
default_threads: int = get_default_cpu_threads()
seed_max: int = 2 ** 32 - 1


def frantic_search(dictionary: Dict[Hashable, Any],
                   *possible_keys: Hashable) -> Any:
    """
    Find first of the several keys that are in a dictionary and return the value.

    :param dictionary: dictionary to search.
    :param possible_keys: any number of possible keys (preferred first).
    :return: dictionary[first_found_key].
    """
    for key in possible_keys:
        if key in dictionary:
            return dictionary[key]
    missed_keys = ', '.join([str(k) for k in possible_keys])
    raise KeyError(f'Found none of the: {missed_keys}')


def total_time(func: Callable) -> Callable:
    """
    Timing decorator logging function execution duration.

    :param func: Function to wrap.
    :return: Wrapped function.
    """
    @wraps(func)
    def timed(*args: Any, **kwargs: Any) -> Any:
        from time import time
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        duration = end_time - start_time
        human_readable = (f'{int(duration // 3600):02d}:'
                          f'{int((duration % 3600) // 60):02d}:'
                          f'{int(duration % 60):02d}')
        logger.info(f'Function "{func.__name__}" executed in {human_readable} (hh:mm:ss)')
        return result

    return timed


def checkpoint(funct: Callable) -> Callable:
    """
    Serialization checkpoint decorator that caches function results on disk.

    :param funct: Function to wrap.
    :return: Wrapped function with checkpoint disk caching.
    """
    signature = inspect.signature(funct)

    @wraps(funct)
    def save_checkpoint(*args: Any, **kwargs: Any) -> Any:
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


def timeout_signal(seconds: int = 1800) -> Callable:
    """
    Signal-based timeout decorator for POSIX systems.

    :param seconds: Maximum allowed runtime in seconds before raising TimeoutError.
    :return: Wrapped decorator function.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            def _handle_timeout(signum: int, frame: Any) -> None:
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds.")

            old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


def _parallel_worker_dispatch(shared_objects: Tuple[Callable, Dict[str, Any], Any],
                              item: Any) -> Any:
    """
    Top-level picklable target dispatcher compatible with forkserver and spawn.

    :param shared_objects: Tuple of (target_function, user_kwargs, user_shared_objects).
    :param item: Single input argument yielded for the task.
    :return: Target function evaluation result.
    """
    target_fn, kwargs, user_shared = shared_objects
    if user_shared is not None:
        return target_fn(user_shared, item, **kwargs)
    return target_fn(item, **kwargs)


class Parallel:
    """
    High-performance multiprocessing wrapper around MPIRE WorkerPool.

    Manages parallel worker pools, input collection mapping, dynamic chunk size calculation,
    and seed generation for replicate runs in bioinformatic processing workflows.
    """

    def __init__(self,
                 parallelized_function: Optional[Callable] = None,
                 input_collection: Optional[Collection] = None,
                 random_replicates: Optional[int] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 n_jobs: Optional[int] = None,
                 description: Optional[str] = None,
                 bar: bool = True,
                 bar_color: str = 'cyan',
                 chunk_size: Any = 'auto',
                 shared_objects: Any = None,
                 keep_order: bool = True,
                 function: Optional[Callable] = None,
                 **extra_kwargs: Any) -> None:
        """
        Initialise Parallel instance and execute worker pool mapping.

        :param parallelized_function: Target callable function to execute in parallel workers.
        :param input_collection: Collection of input items to map across worker processes.
        :param random_replicates: Integer count of random replicate iterations to execute.
        :param kwargs: Optional dict of keyword arguments passed to parallelized function.
        :param n_jobs: Integer count of parallel worker jobs (defaults to available - 1).
        :param description: String label displayed on the progress bar.
        :param bar: Boolean flag specifying whether to display a progress bar.
        :param bar_color: Color string for progress bar.
        :param chunk_size: Integer size or 'auto' for dynamic worker task chunking.
        :param shared_objects: Optional shared memory object graph passed to workers.
        :param keep_order: Boolean flag specifying whether to preserve input item ordering.
        :return: None
        """
        target_fn = parallelized_function if parallelized_function is not None else function
        if target_fn is None:
            raise ValueError("Parallel requires a target function (parallelized_function or function).")

        assert bool(random_replicates) ^ bool(input_collection is not None), \
            'You need to specify EITHER an input collection OR number of random replicates'

        n_jobs = get_default_cpu_threads(n_jobs)
        kwargs = {} if not kwargs else kwargs
        description = description if description else getattr(target_fn, "__name__", "Parallel")

        if random_replicates:
            ss = np.random.SeedSequence()
            child_seeds = ss.spawn(random_replicates)
            input_list: Any = [int(s.generate_state(1)[0]) for s in child_seeds]
            description = f'{description} 🎲'
        else:
            input_list = input_collection

        total_items: Optional[int] = None
        if isinstance(input_list, (list, tuple)):
            total_items = len(input_list)
            if chunk_size == 'auto':
                if total_items > 0 and n_jobs > 0:
                    chunk_size = max(1, min(64, total_items // (n_jobs * 4)))
                else:
                    chunk_size = 1
        else:
            if chunk_size == 'auto':
                chunk_size = 32

        worker_kwargs = {}
        # Default start method for WorkerPool: use forkserver if available, else spawn
        if 'start_method' not in extra_kwargs:
            avail = mp.get_all_start_methods()
            worker_kwargs['start_method'] = 'forkserver' if 'forkserver' in avail else 'spawn'
        else:
            worker_kwargs['start_method'] = extra_kwargs.pop('start_method')

        worker_kwargs['shared_objects'] = (target_fn, kwargs, shared_objects)

        extra_kwargs.pop('total', None)
        iterable_len = extra_kwargs.pop('iterable_len', None)
        if iterable_len is None and total_items is not None:
            iterable_len = total_items

        def _arg_feeder(source_iter: Any) -> Generator[Tuple[Any], None, None]:
            for item in source_iter:
                yield (item,)

        with WorkerPool(n_jobs=n_jobs, **worker_kwargs) as pool:
            pbar_opts = {'desc': description, 'colour': bar_color} if bar else {}
            if bar and 'unit' in extra_kwargs:
                pbar_opts['unit'] = extra_kwargs.pop('unit')

            map_kwargs: Dict[str, Any] = {
                'progress_bar': bar,
                'progress_bar_options': pbar_opts if bar else None,
                'chunk_size': chunk_size,
                'iterable_len': iterable_len,
            }

            map_func = pool.map if keep_order else pool.map_unordered
            self.result: List[Any] = map_func(_parallel_worker_dispatch,
                                              _arg_feeder(input_list),
                                              **map_kwargs)
        self.results = self.result


def shell(command: List[str],
          stdout: str = 'suppress',
          stdin: Optional[bytes] = None,
          timeout: Optional[float] = None) -> Optional[bytes]:
    """
    Run external CLI commands, ensuring all sub-processes in the process
    group are fully terminated on timeout or failure.

    :param command: List of command arguments.
    :param stdout: Output handling ('suppress', 'capture', or None).
    :param stdin: Optional input bytes.
    :param timeout: Execution timeout threshold in seconds.
    :return: Captured bytes output if stdout == 'capture', else None.
    """
    sanitized_command = [str(c) for c in command]
    logger.info(" ".join(sanitized_command))

    if stdout == 'suppress':
        out_stream, err_stream = subprocess.DEVNULL, subprocess.DEVNULL
    elif stdout == 'capture':
        out_stream, err_stream = subprocess.PIPE, subprocess.PIPE
    else:
        out_stream, err_stream = None, None

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
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        process.wait()
        raise TimeoutError(f'Command "{" ".join(sanitized_command)}" timed out after {timeout} seconds.')

    if process.returncode != 0:
        err_msg = err.decode('utf-8', errors='replace') if err else "Process failed silently"
        raise ChildProcessError(
            f'Command "{" ".join(sanitized_command)}" exited with code {process.returncode}:\n{err_msg}')

    return out if stdout == 'capture' else None


def run_external(command: Union[Sequence[Any], Sequence[Sequence[Any]]],
                 output_file: Optional[Union[str, Path]] = None,
                 stdout: Optional[Union[str, Path, Any]] = "suppress",
                 stdin: Optional[Union[bytes, str, Path, Any]] = None,
                 timeout: Optional[float] = None,
                 text: bool = True,
                 check: bool = True) -> Optional[Union[str, bytes]]:
    """
    Execute external process or piped pipeline with stream file writing or output capture.

    :param command: Sequence of arguments, or sequence of command sequences for pipeline.
    :param output_file: Optional target file path to stream standard output directly into.
    :param stdout: Execution output mode ('suppress', 'capture', file path/handle, or None).
    :param stdin: Optional bytes, string, file path, or input stream passed to stdin.
    :param timeout: Optional float timeout threshold in seconds.
    :param text: Boolean flag specifying whether stream inputs/outputs are text strings.
    :param check: Boolean flag specifying whether non-zero exit codes raise ChildProcessError.
    :return: Output string or bytes if stdout='capture', otherwise None.
    """
    if not command:
        raise ValueError("Command sequence cannot be empty.")

    if isinstance(command[0], (list, tuple)):
        pipeline_commands: List[List[str]] = [
            [str(arg) for arg in cmd] for cmd in command  # type: ignore[union-attr]
        ]
    else:
        pipeline_commands = [[str(arg) for arg in command]]  # type: ignore[union-attr]

    if len(pipeline_commands) == 1:
        logger.info(" ".join(pipeline_commands[0]))
    else:
        logger.info(" | ".join(" ".join(cmd) for cmd in pipeline_commands))

    opened_output_file_handle = None
    if output_file is not None:
        target_path = Path(output_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        opened_output_file_handle = target_path.open("w" if text else "wb")
        stdout_target: Any = opened_output_file_handle
    elif isinstance(stdout, (str, Path)) and stdout not in ("suppress", "capture"):
        target_path = Path(stdout)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        opened_output_file_handle = target_path.open("w" if text else "wb")
        stdout_target = opened_output_file_handle
    elif stdout == "suppress":
        stdout_target = subprocess.DEVNULL
    elif stdout == "capture":
        stdout_target = subprocess.PIPE
    else:
        stdout_target = stdout

    opened_input_file_handle = None
    input_data: Optional[Union[str, bytes]] = None

    if isinstance(stdin, Path):
        opened_input_file_handle = stdin.open("r" if text else "rb")
        stdin_target: Any = opened_input_file_handle
    elif isinstance(stdin, (bytes, str)):
        input_data = stdin
        stdin_target = subprocess.PIPE
    else:
        stdin_target = stdin

    pipeline_processes: List[subprocess.Popen] = []
    total_commands_count = len(pipeline_commands)

    try:
        for command_index, cmd_args in enumerate(pipeline_commands):
            if command_index == 0:
                current_stdin = stdin_target
            else:
                current_stdin = pipeline_processes[-1].stdout

            if command_index == total_commands_count - 1:
                current_stdout = stdout_target
            else:
                current_stdout = subprocess.PIPE

            try:
                process_handle = subprocess.Popen(cmd_args,
                                                  stdin=current_stdin,
                                                  stdout=current_stdout,
                                                  stderr=subprocess.PIPE,
                                                  text=text)
            except FileNotFoundError as file_err:
                binary_executable_name = cmd_args[0] if cmd_args else "Command"
                raise RuntimeError(
                    f"'{binary_executable_name}' binary not found in system PATH."
                ) from file_err

            if command_index > 0 and pipeline_processes[-1].stdout is not None:
                pipeline_processes[-1].stdout.close()

            pipeline_processes.append(process_handle)

        intermediate_stderr_outputs: List[List[str]] = [
            [] for _ in range(total_commands_count - 1)
        ]
        stderr_drain_threads: List[threading.Thread] = []

        for proc_idx in range(total_commands_count - 1):
            proc = pipeline_processes[proc_idx]
            out_list = intermediate_stderr_outputs[proc_idx]

            def _drain_stderr(pipe_proc: subprocess.Popen, storage: List[str]) -> None:
                if pipe_proc.stderr is not None:
                    try:
                        err_content = pipe_proc.stderr.read()
                        if err_content:
                            storage.append(err_content)
                    except Exception:
                        pass

            drain_thread = threading.Thread(target=_drain_stderr,
                                            args=(proc, out_list),
                                            daemon=True)
            drain_thread.start()
            stderr_drain_threads.append(drain_thread)

        if total_commands_count == 1:
            last_process = pipeline_processes[-1]
            captured_stdout, last_stderr = last_process.communicate(input=input_data,
                                                                    timeout=timeout)
        else:
            if input_data is not None and pipeline_processes[0].stdin is not None:
                first_proc_stdin = pipeline_processes[0].stdin

                def _write_stdin_async(proc: subprocess.Popen,
                                       pipe: Any,
                                       data: Union[str, bytes],
                                       is_text_mode: bool) -> None:
                    try:
                        if is_text_mode and isinstance(data, bytes):
                            pipe.write(data.decode("utf-8"))
                        elif not is_text_mode and isinstance(data, str):
                            pipe.write(data.encode("utf-8"))
                        else:
                            pipe.write(data)
                        pipe.flush()
                    except OSError:
                        pass
                    finally:
                        pipe.close()
                        proc.stdin = None

                writer_thread = threading.Thread(target=_write_stdin_async,
                                                args=(pipeline_processes[0],
                                                      first_proc_stdin,
                                                      input_data,
                                                      text),
                                                daemon=True)
                writer_thread.start()

            last_process = pipeline_processes[-1]
            captured_stdout, last_stderr = last_process.communicate(timeout=timeout)

        for drain_thread in stderr_drain_threads:
            drain_thread.join(timeout=timeout)

        error_messages_list: List[str] = []

        for proc_index, process in enumerate(pipeline_processes):
            process.wait(timeout=timeout)
            if proc_index < total_commands_count - 1:
                proc_stderr = (intermediate_stderr_outputs[proc_index][0]
                               if intermediate_stderr_outputs[proc_index] else "")
            else:
                proc_stderr = last_stderr if last_stderr else ""

            if process.returncode != 0 and check:
                cmd_str = " ".join(pipeline_commands[proc_index])
                stderr_text = proc_stderr.strip() if proc_stderr else ""
                error_detail = f'"{cmd_str}" crashed with exit code {process.returncode}'
                if stderr_text:
                    error_detail += f":\n{stderr_text}"
                error_messages_list.append(error_detail)

        if error_messages_list and check:
            raise ChildProcessError("\n".join(error_messages_list))

        if stdout == "capture":
            return captured_stdout
        return None

    except subprocess.TimeoutExpired:
        for process in pipeline_processes:
            process.kill()
            process.wait()
        raise
    finally:
        if opened_output_file_handle is not None:
            opened_output_file_handle.close()
        if opened_input_file_handle is not None:
            opened_input_file_handle.close()
# fmt: on
