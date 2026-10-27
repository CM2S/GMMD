"""
Module running a function in a process of its own.

Gmsh can end the process it runs in rather than raise, which no except catches, and the
run it was part of ended with it. A function of a module that calls gmsh is decorated
with `in_own_process`, and every call to it then runs in a process of its own: what it
returns and what it raises come back to the one that called it, and a process that
ends before it returned is told with an error.

The process is a fresh interpreter, started with `subprocess` on `geommicgen._worker`,
which imports the module of the function and nothing else: the script the calling
process was started from is never run in it, whether or not its top level is kept from
running when imported.
"""

import functools
import importlib
import os
import pickle
import subprocess
import sys
import traceback

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.errors.error_classes import ProcessDied, RaisedInAnotherProcess


def serve():
    """
    Call the function a process was started for, and send back what came of it.

    The call is read from stdin, and what comes of it is sent on stdout, pickled: what
    the function reports, then what it returns or what it raised.
    """
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    # The calling process reads what is sent on stdout, so what is printed here, by
    # Python or by a library it calls, goes to stderr instead. A pipe of its own would
    # leave stdout alone, but handing a process one more pipe is not portable to Windows

    def send(*message):
        channel.write(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL))
        channel.flush()
        # Pickled whole before it is written, so that a message that cannot be pickled
        # leaves nothing of itself on the channel

    module, name, args, kwargs, reporting = pickle.load(sys.stdin.buffer)
    if reporting:
        kwargs = dict(kwargs, report=lambda *reported: send("report", reported))
    try:
        function = getattr(importlib.import_module(module), name).in_this_process
        result = function(*args, **kwargs)
    except Exception as error:  # pylint: disable=broad-except
        trace = traceback.format_exc()
        try:
            send("error", error, trace)
        except Exception:  # pylint: disable=broad-except
            send("error", RuntimeError(str(error)), trace)
        # An error that cannot be pickled is sent as its message
    else:
        send("result", result)


def in_own_process(task):
    """
    Make every call to a function of a module run in a process of its own.

    The function is still called with the same arguments, which are pickled to reach
    the other process, and gives what it returns, pickled back. A keyword *report*, a
    callable, is called in this process with whatever the function calls it with there.
    The function itself stays at *in_this_process* on what this returns.

    Parameters
    ----------
    task: str
        What the function does, for the error a process that ends is told by.

    Returns
    -------
    callable
        The decorator.

    Raises
    ------
    ProcessDied:
        From a call, if the process ended before the function returned.
    """

    def decorate(function):
        @functools.wraps(function)
        def call(*args, **kwargs):
            report = kwargs.pop("report", None)
            process = subprocess.Popen(
                [sys.executable, "-m", "geommicgen._worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
            )
            # The process searches where this one does, so that it imports the
            # package, and the module of the function, from where this one did
            message = None
            try:
                with process.stdin:
                    pickle.dump(
                        (
                            function.__module__,
                            function.__qualname__,
                            args,
                            kwargs,
                            report is not None,
                        ),
                        process.stdin,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                while True:
                    message = pickle.load(process.stdout)
                    if message[0] != "report":
                        break
                    report(*message[1])
            except (BrokenPipeError, EOFError, pickle.UnpicklingError):
                message = None
                # The process ended before it read the call, or before it sent all of
                # what came of it
            except BaseException:
                process.kill()
                raise
            finally:
                process.stdout.close()
                process.wait()
            if message is None:
                raise ProcessDied(task, process.returncode)
            if message[0] == "error":
                raise message[1] from RaisedInAnotherProcess(message[2])

            return message[1]

        call.in_this_process = function
        return call
        # The other process finds the function by its name in its module, which is
        # this wrapper, and calls what it wraps

    return decorate
