# fmt: off
"""
Multi-threading, system CPU thread detection, command execution, and logging utilities.

This module provides system thread count determination, an MPIRE wrapper for parallel process pool
execution, external process invocation, and loguru warning redirection.
"""

# built-ins
import os
import subprocess
import sys
import threading
import warnings
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Union

# standard libraries
import numpy as np

# internal repository/package imports
from loguru import logger
from mpire import WorkerPool


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

        if isinstance(input_list, (list, tuple)):
            total_items = len(input_list)
            if chunk_size == 'auto':
                if total_items > 0 and n_jobs > 0:
                    chunk_size = max(1, min(200, total_items // (n_jobs * 4)))
                else:
                    chunk_size = 1
        else:
            if chunk_size == 'auto':
                chunk_size = 1

        worker_kwargs = {'shared_objects': shared_objects} if shared_objects else {}

        def _worker_wrapper(*args: Any) -> Any:
            item = args[0] if len(args) == 1 else args
            return target_fn(item, **kwargs)

        extra_kwargs.pop('total', None)
        extra_kwargs.pop('iterable_len', None)

        with WorkerPool(n_jobs=n_jobs, **worker_kwargs) as pool:
            pbar_opts = {'desc': description, 'colour': bar_color} if bar else {}
            if bar and 'unit' in extra_kwargs:
                pbar_opts['unit'] = extra_kwargs.pop('unit')

            map_kwargs: Dict[str, Any] = {
                'progress_bar': bar,
                'progress_bar_options': pbar_opts if bar else None,
                'chunk_size': chunk_size
            }

            map_func = pool.map if keep_order else pool.map_unordered
            self.result: List[Any] = map_func(_worker_wrapper,
                                              input_list,
                                              **map_kwargs)
        self.results = self.result


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
