"""
Module running a function in a process of its own.

Gmsh can end the process it runs in rather than raise, which no except catches, and the
run it was part of ended with it. A function of a module that calls gmsh is decorated
with `in_own_process`, and every call to it then runs in a process of its own: what it
returns and what it raises come back to the one that called it, and a process that
ends before it returned is told with an error.
"""

import contextlib
import functools
import importlib
import multiprocessing
import sys
import traceback

# pylint: disable=import-error
# pylint: disable=relative-beyond-top-level
from geommicgen.errors.error_classes import ProcessDied, RaisedInAnotherProcess

START_METHOD = (
    "forkserver"
    if "forkserver" in multiprocessing.get_all_start_methods()
    else "spawn"
)
# How the process is started: forked from a server process that has imported the
# modules below and never run gmsh, or, where there is no such server, started afresh.
# Either way it is sent the arguments. A process forked from the one making the call
# deadlocked once gmsh had run in that one, whose threads OpenCASCADE leaves behind

PRELOADED = [
    "geommicgen.meshing.gmsh_mesher",
    "geommicgen.postproc.plotfuncs.plotting_functions",
]
# Modules the server imports once, so that a process forked from it does not import
# them again. Only the first call starts the server, so every call names all of them


@contextlib.contextmanager
def main_module_hidden():
    """
    Keep a process started meanwhile from running the main module of this one.

    A process that is not forked from this one runs the file this one was started
    from, to find what was defined there, and a script that meshes at its top level,
    with nothing to keep it from running when imported, meshed again in it, or was
    refused. Nothing a call here sends is defined there, so the file and the spec that
    lead to it are taken off the main module while the process is started.
    """
    main_module = sys.modules["__main__"]
    hidden = {
        i_name: vars(main_module).pop(i_name)
        for i_name in ("__file__", "__spec__")
        if i_name in vars(main_module)
    }
    main_module.__spec__ = None
    # Read as an attribute, which every module has, so it is left at None
    try:
        yield
    finally:
        del main_module.__spec__
        vars(main_module).update(hidden)


def call_in_child(connection, module, name, args, kwargs, reporting):
    """
    Call a decorated function in the process this runs in, and send what came of it.

    Parameters
    ----------
    connection: `multiprocessing.connection.Connection`
        End of the pipe the messages are sent through.

    module: str
        Name of the module the function belongs to.

    name: str
        Name of the function in its module.

    args: tuple
        Positional arguments of the call.

    kwargs: dict
        Keyword arguments of the call.

    reporting: bool
        Whether the function is given a *report* that sends what it reports.
    """
    if reporting:

        def report(*reported):
            connection.send(("report", reported))

        kwargs = dict(kwargs, report=report)
    try:
        function = getattr(importlib.import_module(module), name).in_this_process
        result = function(*args, **kwargs)
    except Exception as error:  # pylint: disable=broad-except
        trace = traceback.format_exc()
        try:
            connection.send(("error", error, trace))
        except Exception:  # pylint: disable=broad-except
            connection.send(("error", RuntimeError(str(error)), trace))
        # An error that cannot be pickled is sent as its message
    else:
        connection.send(("result", result))
    connection.close()


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
            context = multiprocessing.get_context(START_METHOD)
            if START_METHOD == "forkserver":
                context.set_forkserver_preload(PRELOADED)
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=call_in_child,
                args=(
                    sender,
                    function.__module__,
                    function.__qualname__,
                    args,
                    kwargs,
                    report is not None,
                ),
            )
            with main_module_hidden():
                process.start()
            sender.close()
            message = None
            try:
                while True:
                    message = receiver.recv()
                    if message[0] != "report":
                        break
                    report(*message[1])
            except EOFError:
                message = None
                # The pipe was closed with nothing more on it, which only happens when
                # the process ended before it could send what came of the call
            except BaseException:
                process.terminate()
                raise
            finally:
                receiver.close()
                process.join()
            if message is None:
                raise ProcessDied(task, process.exitcode)
            if message[0] == "error":
                raise message[1] from RaisedInAnotherProcess(message[2])

            return message[1]

        call.in_this_process = function
        return call
        # The other process finds the function by its name in its module, which is
        # this wrapper, and calls what it wraps

    return decorate
