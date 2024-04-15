import json
from xmlrpc.server import SimpleXMLRPCServer
from queue import Queue
import threading
import time
from datetime import datetime, timedelta
import os
import importlib as il
from importlib import util as ilutil
from importlib import abc as ilabc
import tokenize
import inspect
import sys

import linker

exit_threads = False


def check_argspec_valid(funcSpec: tuple, *somewhereArgs, acceptVariadic=True):
    args: list = funcSpec[0]
    varargs: str = funcSpec[1]
    varkw: str = funcSpec[2]
    varargdefaults = funcSpec[3]
    kwonlyargs: list = funcSpec[4]
    kwonlydefaults = funcSpec[5]
    annotations = funcSpec[6]
    if (varargs is not None or varkw is not None) and acceptVariadic:
        # if variadic is permitted, and variadic args are present, allow default.
        return True, []
    errors:list[str] = []
    valid = True
    for v in somewhereArgs:
        if v not in args and v not in kwonlyargs:
            # param isn't present and we can't accept variadic
            errors.append(f"`{v}` not in function {args=} or {kwonlyargs=}")
            valid = False
    return valid, errors


class Parameters:
    P_NUM_THREADS = 5
    P_MAX_PENDING_REQUESTS = 500
    P_MAX_AWAIT_MINS_TIMEOUT = 10
    P_LINKER_PATH = "./linker.py"  # use forward slashes regardless, and relative path if needed.
    # Should be ~/Ant_Project_TF2/linker.py in production settings for ubuntu linux

    def __init__(self):
        self.P_NUM_THREADS = self.P_NUM_THREADS
        self.P_MAX_PENDING_REQUESTS = self.P_MAX_PENDING_REQUESTS
        self.P_MAX_AWAIT_MINS_TIMEOUT = self.P_MAX_AWAIT_MINS_TIMEOUT
        self.P_LINKER_PATH = os.path.abspath(self.P_LINKER_PATH.replace("/", os.sep))

    def validate(self):
        if not os.path.exists(self.P_LINKER_PATH):
            print(f"ERROR: Linker.py as specified `{self.P_LINKER_PATH}` does not exist. Exiting...")
            exit(-1)

        # check validity of file at linker path

        if os.path.splitext(self.P_LINKER_PATH)[1] != ".py":
            print(f"ERROR: Linker file at `{self.P_LINKER_PATH}` does not end in `.py`")
            exit(-1)

        # check if this file has two methods of proper signature
        print(f"Getting Spec for Linker Module at `{self.P_LINKER_PATH}`...")
        self.__linker_spec__ = ilutil.spec_from_file_location("__main__", self.P_LINKER_PATH)
        self.__linker_module__ = ilutil.module_from_spec(self.__linker_spec__)
        # this appears to be a global assignment, so exportation isn't needed (assumption)
        sys.modules["linker"] = self.__linker_module__
        print("Executing Linker Code...")
        self.__linker_spec__.loader.exec_module(self.__linker_module__)

        dirspec = dir(self.__linker_module__)
        if "add_parameters" not in dirspec or "do_classification" not in dirspec:
            print("ERROR: Linker File is missing function signatures: ", end="")
            if "add_parameters" not in dirspec:
                print(f"'add_parameters(identity, filepath)'")
            if "do_classification" not in dirspec:
                print(f"'do_classification(identity, filepath)'")
            exit(-1)
        print("Checking functional signatures....")
        # check functional signatures
        do_classification_sig = inspect.getfullargspec(linker.do_classification)
        add_parameters_sig = inspect.getfullargspec(linker.add_parameters)

        valid_class, err_c = check_argspec_valid(do_classification_sig, "filepath", "identity")
        valid_params, err_p = check_argspec_valid(add_parameters_sig, "filepath", "identity")

        if not valid_class or not valid_params:
            print(f"ERROR: Functional Specification invalid for `{self.P_LINKER_PATH}`:")
            if err_c:
                print(f"\tdo_classification missing parameters: \n\t\t" + ",\n\t\t".join(err_c))
            if err_p:
                print(f"\tadd_parameters missing parameters: \n\t\t" + ",\n\t\t".join(err_p))
            exit(-1)

        print("Functional Signatures valid!")
        return self


    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        params = cls()
        for line in lines:
            key, value = line.strip().split('=')
            setattr(params, key, eval(value))

        return params

    @classmethod
    def instantiate_from_file_or_defaults(cls, filename):
        if os.path.exists(filename):
            if not open(filename, 'r').readlines():
                params = Parameters()
                params.to_file(filename)

            return Parameters.from_file(filename).validate()
        else:
            params = cls()
            # initialize from new instance
            Parameters.to_file(params, filename)
            return Parameters.from_file(filename).validate()

    def to_file(self, filename):
        with open(filename, 'w') as file:
            for attr, value in vars(self).items():
                if attr.startswith("P_"):
                    file.write(f"{attr}={repr(value)}\n")

    def __str__(self):
        o = [f"{attr}={repr(value)}" for attr, value in vars(self).items() if "P_" in attr]
        return "\n".join(o)

    def get_as_list(self):
        return [f"{attr}={repr(value)}" for attr, value in vars(self).items() if "P_" in attr]


class Request:
    id: str
    filepath: str
    result: dict
    createdat: datetime = datetime.now()
    finished: bool = False

    def __init__(self, filepath, identity):
        self.filepath = filepath
        self.id = identity
        self.result = {}

    def set_result(self, resdict, *, set_finished=True):
        self.result = resdict
        if set_finished:
            self.finished = True
        return None

    @property
    def out_of_date(self):
        return (datetime.now() - self.createdat) > timedelta(instanceParameters.P_MAX_AWAIT_MINS_TIMEOUT)


pending_requests: Queue = Queue()  # constrained by max pending requests - will drop if insufficient resources available.
completed_requests: list[Request] = []
invalidated: list[str] = []

def do_classification_threaded(x):
    # as ele is an object (non-primitive), it is passed by reference here
    global pending_requests
    global completed_requests
    print(f"Initializing Classification Thread; Id={x}")
    while True:
        if exit_threads:
            break
        if pending_requests.qsize() > 0:
            req: Request = pending_requests.get()

            print(f"{x:<3}| Classifying Req {req.id}")
            results: dict = linker.do_classification(req.id, req.filepath)
            req.result.update(results)
            completed_requests.append(req)
    print(f"Classification Thread {x} Terminated!")


def setup_queue_handler():
    # setup lists to populate with None's
    global instanceParameters
    # note here that active requests MUST BE CONSTRAINED to the NUM_THREADS as having infinitely many classifications
    # going on simultaneously will deplete server resources
    # As each request is processed in another thread, the value will hang as Request(), and when complete, the function
    # will return None (see method Request.set_result)
    threads = []
    for x in range(instanceParameters.P_NUM_THREADS):
        thread = threading.Thread(target=do_classification_threaded, args=(x,))
        thread.start()
        threads.append(thread)
    return threads


def handle_request_timeout_thread():
    global completed_requests
    global pending_requests
    global invalidated
    while True:
        if exit_threads:
            break
        invalidated = [req.id for req in completed_requests if req.out_of_date]

        completed_requests = [req for req in completed_requests if not req.out_of_date]
        # once requests are added to the queue, it becomes immutable while its there, so we just let the queue process it out or cancel with exit_threads.
        time.sleep(1)

    print("Request Timeout Handler Thread Terminated")


class RPCMethods:
    # As required, add more things here...
    @staticmethod
    def do_classification(filepath, identity):
        global pending_requests
        global instanceParameters
        print(f"New Request: {filepath}, {identity}")
        # attempt to add to queue, if space permits
        if pending_requests.qsize() >= instanceParameters.P_MAX_PENDING_REQUESTS:
            return False, {"reason": "Too Many Requests"}
        pending_requests.put(Request(filepath, identity), block=False)
        return True,  {"message":"Added to Classification Queue Successfully!"}
        #return (True, HTTPMessages.Ok.value.set_params(
        #    reason="Note: Not Implemented. This is a stub, but for testing purposes, this can be adjusted.", image=None, ).get_dict())

    @staticmethod
    def check_classification(identity) -> [bool, dict]:
        """
        Returns True if complete, and False if not. The dictionary element will update the Backend dict to return.
        :param self:
        :param identity:
        :return:
        """
        global invalidated
        matches: list[Request] = [r for r in completed_requests if r.id == identity]
        if len(matches) == 0:
            if identity in invalidated:
                invalidated = [ident for ident in invalidated if not ident==identity]
                return False, {"message": "Gone: Timeout reached: Deleted.", "status": 410}
            return False, {"message":"Too Early", "status":425}
        else:
            req: Request = matches[0]
            req.result.update(linker.add_parameters(req.id, req.filepath))
            return True, req.result

    @staticmethod
    def trigger_exits():
        global exit_threads
        print("Thread Exiter Triggered!")
        exit_threads = True


def main():
    global instanceParameters
    filename = os.path.join(os.getcwd(), "antai_rpc.config")
    print(f"Initializing from Config - location: {filename}")
    instanceParameters = Parameters.instantiate_from_file_or_defaults(filename)

    print("Initializing Queue Handlers...")
    #queue_thread = threading.Thread(target=setup_queue_handler)
    #queue_thread.start()
    setup_queue_handler()
    # run timeout thread
    timeout_thread = threading.Thread(target=handle_request_timeout_thread)
    timeout_thread.start()

    print("Serving RPC...")
    server = SimpleXMLRPCServer(('localhost', 3000), allow_none=True, use_builtin_types=True)
    server.register_instance(RPCMethods())
    server.serve_forever()


if __name__ == '__main__':
    instanceParameters = None
    main()
