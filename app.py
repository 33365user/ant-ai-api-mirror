import logging
import inspect
from io import TextIOWrapper

from werkzeug.datastructures import FileStorage

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from flask import Flask, render_template, flash, redirect, url_for, send_from_directory, jsonify, \
    render_template_string, make_response

from flask_restful import Api, Resource, reqparse, request
from flask_restful_swagger import swagger

import json
import os, sys
from params import Params
from utils import *
from loggermod import *
from werkzeug.utils import secure_filename
import HTTPStatusMessages as HSM
import threading
import datetime
from datetime import timedelta
from datetime import datetime as dt
from xmlrpc.client import ServerProxy
from queue import Queue, PriorityQueue
import time
from waitress import serve
import markdown
import pandas as pd
import hashlib
import csv
print(sys.version_info)

@log_it(loggerName="setup")
def setup(params: Params, app):
    dirs = [params.UPLOAD_BASE_DIRECTORY, params.IMAGE_STORAGE_SAVE_PATH, params.CSV_DIR_PATH]
    for d in dirs:
        if not check_relative_path_exists(d):

            if not ensure_folder_exists(d):
                raise OSError(f"Unable to Establish `{d}`. Please Check Permissions or file structure.")

    app.config['UPLOAD_FOLDER'] = params.UPLOAD_BASE_DIRECTORY
    app.config['MAX_CONTENT_LENGTH'] = params.MAX_UPLOAD_FILESIZE
    app.config['FILESTORAGE_FOLDER'] = params.IMAGE_STORAGE_SAVE_PATH
    app.config['FILESTORAGE_DOSAVE'] = params.SAVE_IMAGES_AFTER_CLASSIFICATION


def handle_queue():
    global todoq
    global finishedq
    global do_exit
    """
    Used by an alternate thread to process the Queues and handle them appropriately
    :return:
    """
    # todo: make this allow for several AI instances to speed up analysis.
    previous_queue_size = 0
    print("Initializing Queue Handler...")
    log_line("Initializing Queue Handler...", loggername="queuehandler")
    while True:
        try:
            with (ServerProxy('http://localhost:3000/',
                             allow_none=True, use_datetime=True, use_builtin_types=True) as proxy):
                while True:
                    if do_exit:
                        log_line("Exiting Queue Handler Main Loop due to `do_exit=True`", loggername="queuehandler")
                        break

                    # check here for the finished list and remove elements more than QUEUE_TIMEOUT old: prevent denial of service.
                    now = datetime.datetime.now()

                    # if a request has been completed, and is waiting on the queue for more time than it is intended, we delete it.
                    outofdate: list[ClassificationRequest] = [e for e in finishedq if e.out_of_date]
                    for e in outofdate:
                        print(f"Request {e.identity} has timed out. Deleting information...")
                        try:
                            os.remove(e.filepath)
                        except OSError:
                            print("Unable to delete file...")
                    del outofdate
                    finishedq = [e for e in finishedq if not e.out_of_date]

                    if todoq.qsize() != previous_queue_size:
                        print("Size of Queue: " + str(todoq.qsize()))
                        previous_queue_size = todoq.qsize()
                    if todoq.qsize() == 0:
                        continue

                    ele: ClassificationRequest = todoq.get(block=False)
                    if (datetime.datetime.now() - ele.createdat) > Params.QUEUE_TIMEOUT:
                        # we want to delete it, not process it: its taking up space.
                        # as its already off the todoq at this point, we just don't process it further this iteration - effectively deleted
                        log_line(f"Deleting Image corresponding to {ele.identity}: Timeout reached for processing...", loggername="queuehandler",
                                 level=LogLevel.L_DEBUG)
                        try:
                            os.remove(ele.filepath)
                        except OSError:
                            log_line(f"\t Could not remove file `{ele.filepath}` for some reason.")
                        del ele
                        continue
                    # todo: update ele.details rather than set: to allow tunnelling of authentication and other details from client.
                    if ele.status == 0:  # Waiting to be processed.
                        if not os.path.exists(ele.filepath):  # race condition handling
                            print("Request is awaiting OS File creation.")
                            todoq.put(ele)  # put it  back on the queue for later processing.
                            continue
                            # todo - update for new structure of RPC Server.
                        isprocessing, message = proxy.do_classification(ele.filepath, ele.identity)
                        if isprocessing:
                            ele.status = 1  # Currently Processing
                            # there are details to add
                            ele.details = dict(active=True, reason="Classification Progressing...")
                            ele.details.update(message)
                            todoq.put(ele)  # we need to keep track of this until its finished!
                            continue
                        else:
                            ele.status = 3  # Aborted
                            # no point in adding details, the error was server-side.
                            ele.details = dict(active=False, reason="Classification was aborted due to technical error server side.", code=1)
                            ele.details.update(message)
                            finishedq.append(ele)
                            continue

                    elif ele.status == 1:  # currently being processed
                        # prevent spurious checks (multiple within 1 second) - as it will exhaust resources.
                        if (datetime.datetime.now() - ele.checkedat) < Params.API_QUERY_AWAIT:
                            # send it back to the todoq
                            todoq.put(ele)
                            continue
                        # check with AntAI module regarding status of classification.
                        complete, message = proxy.check_classification(ele.identity)
                        if complete:  # Finished Inference of Some Description
                            ele.details.update(message)
                            ele.details.update(dict(active=False))
                            ele.status = 2
                            finishedq.append(ele)
                        else:  # Too early
                            todoq.put(ele)
                        continue

                    elif ele.status > 1:  # this should be impossible, but if it is, we add it to finished anyway rather than dropping it.
                        ele.details.update(dict(code=-1, error=f"Unknown source for Status: {ele.status} - Improper Source"))
                        finishedq.append(ele)

            if do_exit:
                log_line("Exiting Queue Handler Thread due to `do_exit=True`", loggername="queuehandler")
                break
        except BaseException as BE:
            exc = sys.exc_info()
            tr = traceback.format_exc()
            log_line(f"Error while handling Queue Handler: {str(exc[0].__name__)}\n\t{tr}", level=LogLevel.L_ERROR, loggername="queuehandler")
            print(f"Error while handling Queue Handler: {str(exc[0].__name__)}\n\t{tr}")
            log_line("Lost Connection to the Server Proxy. Retrying in 30 Seconds...", loggername="queuehandler",
                     level=LogLevel.L_WARNING)
            time.sleep(30)

    log_line("Unusual Scoped Access to Queue Handler Thread  - Either Exiting or unknown condition triggered.",
             loggername="queuehandler", level=LogLevel.L_WARNING)

do_exit = False
SPEC_URL = "/api/spec"
todoq = Queue(Params.MAX_AI_QUEUE_SIZE)
finishedq = []

make_logging(Params.DB_LOG_DIR_PATH, Params.DB_LOG_LEVEL, Params.DB_LOG_FORMAT)
print("Activating Queue Handler Thread")
queuehandler_thread = threading.Thread(target=handle_queue)
queuehandler_thread.start()
app = Flask(__name__)
api = swagger.docs(Api(app), apiVersion=Params.API_VERSION, api_spec_url=SPEC_URL)
setup(Params(), app)

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'  # this is where you would render the website proper if implemented.


@app.route('/exit')
@log_it(LogLevel.L_INFO, print_results=True, loggerName="admin", other="Debug Method Triggered: {0} - Remember to Remove!")
def exit_queue_thread():
    # todo: note not for production build: this is a kill switch!
    print("Thread Exiter Triggered!")
    global do_exit
    do_exit = True
    try:
        with (ServerProxy('http://localhost:3000/',
                          allow_none=True, use_datetime=True, use_builtin_types=True) as proxy):
            proxy.trigger_exits()
        ret = (HSM.HTTPMessages.Accepted.value.set_params(do_exit=True, aimodule_complied=True).get(),
               HSM.HTTPMessages.Accepted.value.status)
    except:
        ret = (HSM.HTTPMessages.Accepted.value.set_params(do_exit=True, aimodule_complied=False).get(),
               HSM.HTTPMessages.Accepted.value.status)
    return ret


class UploadResource(Resource):
    """Main Client Interaction - Upload Ant Images - set {target}=`classify`; Check Classification Progress / Get Results in `/check_classify/<process id>`"""

    @staticmethod
    @app.route('/upload/<target>', methods=['GET', 'POST'])
    @app.route('/upload', methods=['GET', 'POST'])
    @log_it(LogLevel.L_DEBUG, loggerName="methods", print_results=False)
    def post(target: str = ""):
        if target != "classify":
            return jsonify(FormHTTPMessage(403, "Forbidden", dict(reason="Wrong Endpoint, try /upload/classify"))), 403
        if request.method == 'POST':
            longitude, latitude = request.form.get('longitude'), request.form.get('latitude')
            longitude, latitude = change_type(longitude, float, 0.0), change_type(latitude, float, 0.0)
            log_line(f"{longitude=}:of type {type(longitude)}, {latitude=} of type {type(latitude)}")

            # check if the post request has the file part
            if 'file' not in request.files:
                ret = (HSM.HTTPMessages.Bad_Request.value.set_params(reason="No Media File Attached").get(),
                       HSM.HTTPMessages.Bad_Request.value.status)
                return ret
            file = request.files['file']

            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                ret = (HSM.HTTPMessages.Bad_Request.value.set_params(reason="No Media File Attached").get(),
                       HSM.HTTPMessages.Bad_Request.value.status)
                return ret

            if file and Params.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # secure always by renaming the file
                ext = filename.split('.')[1]
                filename = (generate_random_string(25))
                fname = filename
                filename += f".{ext}"
                fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file: FileStorage = file

                if todoq.qsize() >= Params.MAX_AI_QUEUE_SIZE:
                    # if the queue is full, don't let the request join it at the moment, the AI is busy.
                    ret = ((HSM.HTTPMessages.Too_Many_Requests.value.set_params(reason="Server Busy",
                                                                                hint="Try Again Later")).get(),
                           HSM.HTTPMessages.Too_Many_Requests.value.status)
                    return ret

                def save_to_path(fpath, file_content, fname, longitude, latitude):
                    # Params required despite scope as this is meant for deferred execution (after this method is finished)
                    # context management to assuage permissions manager.
                    with open(fpath, 'wb') as f:
                        f.write(file_content)
                    cr = ClassificationRequest(fname, fpath, decimalLatitude=latitude, decimalLongitude=longitude)
                    todoq.put_nowait(cr)

                file_content = file.read()
                t = threading.Thread(target=save_to_path, args=(fpath, file_content, fname, longitude, latitude))
                t.start()

                ret = ((HSM.HTTPMessages.Created.value.set_params(identity=fname)).get(),
                       HSM.HTTPMessages.Created.value.status)
                # nb: HTTPMessages.Processing (102) is not supported for some reason.
                return ret
                # return redirect(url_for('uploaded_file', filename=filename))
            else:
                return ((HSM.HTTPMessages.Unsupported_Media_Type.
                         value.set_params(fname=file.filename,
                                          range=list(Params.ALLOWED_UPLOAD_EXTENSIONS))).get(),
                        HSM.HTTPMessages.Unsupported_Media_Type.value.status)

        return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <input type=file name=file>
          <input type=number name=latitude placeholder="Latitude" step="any">
          <input type=number name=longitude placeholder="Longitude" step="any">
          <input type=submit value=Upload>
        </form>
        '''


@log_it(LogLevel.L_DEBUG, print_results=False, loggerName="methods")
def get_csv_filename_latest_or_create() -> str:
    """
    This helper method will get the filename for a csv to save the current record to.
    If no files exist, it will return today's date.
    If the most recent file is more than 1 week old (as determined by `CSV_MAKE_NEW_DELTA` in Params), then returns today's date.
    Otherwise, returns the most recent csv filename (date) that is less than a week old.
    :return:
    """
    files = [f for f in os.listdir(Params.CSV_DIR_PATH) if f.endswith("csv")]
    selected = None
    now = datetime.datetime.now()
    conv = lambda x,y,z: datetime.datetime(year=int(x), month=int(y), day=int(z), tzinfo=datetime.datetime.now().tzinfo)

    dts = [conv(*(os.path.basename(f) if "." not in os.path.basename(f) else os.path.basename(f).split(".")[0]).split('-')) for f in files]
    dts = sorted(dts)
    if len(dts) == 0:
        selected = None
    else:
        # at least one exists, take the latest
        selected = dts[-1]

    if selected is None:  # if no files exist, we must make one for now.
        df = pd.DataFrame(columns=Params.CSV_COLUMNS)
        df.to_csv(os.path.join(Params.CSV_DIR_PATH, f"{now.year}-{now.month}-{now.day}.csv"), index=False)
        selected = now

    elif (now - selected) > Params.CSV_MAKE_NEW_DELTA:  # if the current file is older than the delta (usually 1 week)
        # make new file to put.

        df = pd.DataFrame(columns=Params.CSV_COLUMNS)
        df.to_csv(os.path.join(Params.CSV_DIR_PATH, f"{now.year}-{now.month}-{now.day}.csv"), index=False)
        selected = now

    #f = open(os.path.join(Params.CSV_DIR_PATH, f"{selected.year}-{selected.month}-{selected.day}.csv"), 'a')
    return str(os.path.join(Params.CSV_DIR_PATH, f"{selected.year}-{selected.month}-{selected.day}.csv"))


@log_it(LogLevel.L_DEBUG, print_results=True, loggerName="methods threaded")
def del_id_from_finished(identity):
    """
    Handles how the system should delete or save information for a classification.
    If the classification result ended in a code != 1, then its invalid to save, as it hasn't been positively classified.
    If it is code==1, then we need to save it to the right CSV file.
    :param identity: identity of record to delete / save
    :return:
    """
    global finishedq
    # check in the uploads folder: delete the file created - also if SAVE_IMAGES_AFTER_CLASSIFICATION==True, then save it to the IMAGE_STORAGE_SAVE_PATH
    ele = [e for e in finishedq if e.identity == identity]
    if not ele:
        raise ValueError("Race Condition excepted in `del_id_from_finished` as element to delete is no longer present.")
    e: ClassificationRequest = ele[0]
    file_name = os.path.basename(e.filepath)
    # we only want to save information for classifications in the positive case - (only the ants that are targeted by the AI)
    # if its a negative detection, then thats useful to know!
    if Params.SAVE_IMAGES_AFTER_CLASSIFICATION and e.details['code'] == 1:
        os.rename(e.filepath, os.path.join(Params.IMAGE_STORAGE_SAVE_PATH, file_name))

        fname = get_csv_filename_latest_or_create()
        e.write_to_csv(fname)

    else:
        os.remove(e.filepath)
    f = [e for e in finishedq if e.identity != identity]
    finishedq = f
    return finishedq


class CheckClassificationResource(Resource):
    """Used to get the details of whether a certain submitted image has been classified or is in the process of being classified."""

    @staticmethod
    @app.route('/check_classify/<identity>', methods=['GET'])
    @app.route('/check_classify/', methods=['GET'])
    @log_it(LogLevel.L_DEBUG, print_results=True, loggerName="methods")
    def get(identity: str = ""):
        if identity == "":
            ret = (HSM.HTTPMessages.Bad_Request.value.set_params(
                reason="Missed <identity> extension to endpoint. Try `/check_classify/{identity}`").get(),
                   HSM.HTTPMessages.Bad_Request.value.status)
            return ret

        # Check if the image is in the directory (meaning it has been set aside for processing)- if its not there, then this call to check classify is wrong
        dirl = os.listdir(Params.UPLOAD_BASE_DIRECTORY)
        file_names_without_extension = [os.path.splitext(file)[0] for file in dirl]
        pos = [f for f in file_names_without_extension if f == identity]
        if not pos:
            return (HSM.HTTPMessages.Not_Found.value.set_params(reason="No File with that identity on system. Please check your identity key.").get(),
                    HSM.HTTPMessages.Not_Found.value.status)

        finished = [e for e in finishedq if e.identity == identity]
        if not finished:
            # search through active queue for processing
            ele = search_queue(todoq, identity)
            if not ele:
                return (HSM.HTTPMessages.Too_Early.value.set_params(reason=f"Classification in Progress").get(),
                        HSM.HTTPMessages.Too_Early.value.status)
        else:
            # return code jsonified - as the process is completed...
            req: ClassificationRequest = finished[0]
            ret = HSM.HTTPMessages.Ok.value.set_params(req.details.copy())
            del_id_from_finished(identity)
            return (ret.get(), ret.status)


def find_file_by_id(folder_path, file_id):
    for filename in os.listdir(folder_path):
        if file_id in os.path.splitext(filename)[0]:
            return filename
    return None

@app.route('/uploads/<filename>')
@log_it(LogLevel.L_DEBUG, print_results=False)
def uploaded_file(filename):
    folder = app.config['FILESTORAGE_FOLDER']
    fname = find_file_by_id(folder, filename)
    filename = fname if fname is not None else filename
    try:
        return send_from_directory(app.config['FILESTORAGE_FOLDER'], filename)
    except:
        return HSM.HTTPMessages.Not_Found.value.set_params(reason="Does not exist.").get(), HSM.HTTPMessages.Not_Found.value.status

@app.route('/policy', methods=['GET'])
@app.route('/policy/', methods=['GET'])
@app.route('/policy/<file>', methods=['GET'])
@app.route('/policy/<file>/', methods=['GET'])
@app.route('/policy/<file>/<mode>', methods=['GET'])
@log_it(LogLevel.L_DEBUG, print_results=False)
def download_policy(file="None", mode="html"):
    policy_path = os.path.join(os.getcwd(), "policy")
    policies_format = """
    <!doctype html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Policies</title>
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: flex;
                margin: 0;
            }
            
            .centered-div {
                text-align: center;
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 10px;
                margin: 10%;
            }
            /* Adjust Format of Common Elements */
            p {
                max-width: 70%;
                margin: 5px auto;
                
            }
            
            /* Adjust the position of list markers */
            ol {
                max-width: 70%; /* Set maximum width to 70% of parent div */
                margin: 0 auto; /* Center the list within its container */
                padding-left: 0; /* Remove default padding */
                counter-reset: my-counter; /* Reset the counter */
                list-style-type: none; /* Remove default list style */
            }
            
            ol li {
                display: flex;
                justify-content: flex-start;
                align-items: center;
                counter-increment: my-counter; /* Increment the counter */
                margin-bottom: 0.5em; /* Optional: Add some spacing between list items */
            }
            
            ol li::before {
                content: counter(my-counter) ". "; /* Use counter value as list marker */
                font-weight: bold; /* Optional: Make the counter bold */
                margin-right: 0.5em; /* Optional: Add some space between counter and text */
                width: 2em; /* Set a fixed width for list markers */
                text-align: right; /* Align numbers to the right */
            }
            
            /* Allow long words to wrap */
            ol li {
                word-break: break-word;
                align-items: center;
            }
            
            
        </style>
    </head>
    <body>
        <div class="centered-div" id="policyhtml">
            { innerhtml }
        </div>
    </body>
    </html>
    """
    other = """
            /* Adjust the position of list markers */
            ol, ul {
                padding-left: 5em; /* Add some padding to the left */
            }
            
            /* Center the content within the list items */
            li {
                text-align: left;
                margin-bottom: 0.5em; /* Optional: Add some spacing between list items */
            }
    """
    if file == "tos":
        path = os.path.join(policy_path, "Terms of Service.md")
    elif file == "pp":
        path = os.path.join(policy_path, "Privacy Policy.md")
    else:
        return """
            <!doctype html>
            <body>
            <h1> Policies Directory </h1>
            <a href="/policy/tos"> Terms of Service </a>
            <br>
            <a href="/policy/pp"> Privacy Policy </a>
            </body>
            """

    with open(path, "r", encoding="utf-8") as input_file:
        text = '\n'.join(input_file.readlines())
    html = markdown.markdown(text)
    html = policies_format.replace("{ innerhtml }", html)
    response = make_response(html)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    if mode == "md":
        return text
    return response


class GetRecordsResource(Resource):
    """ Will be used by ALA to get records from our system. Extension endpoint specified `startdate-enddate` or literal ['weekly', 'monthly', 'quarterly']
        Do note that 'monthly' and 'quarterly' start on the month, rather than in a window around the current date. This means that on 1/11/2023 for instance, only records created 1/11/2023 -> 30/11 will be reported.
        This may also not handle edge cases like records created in the week crossing the monthly or quarterly boundary. Use 'weekly' scope every week or specify more precisely through the available endpoint.
    """
    @staticmethod
    def handle_DDMMYYYY(s: str) -> dt:
        """
        Here, we assume the string is in DDMMYYYY format, and don't error check. This will convert it to a datetime.datetime (timezone aware) object.
        :param s: a string in `DDMMYYYY` format.
        :return:
        """
        d = int(s[:2])
        m = int(s[2:4])
        y = int(s[4:8])
        tzinfo = dt.now().tzinfo
        return dt(year=y, month=m, day=d, tzinfo=tzinfo)

    @staticmethod
    def get_data_df(startdate, enddate):
        """
        This method will fuse all files (.csv) in `CSV_DIR_PATH` into a pandas dataframe and jsonify to export to the client between `startdate` and `enddate`
        :param startdate:
        :param enddate:
        :return:
        """
        files = [f for f in os.listdir(Params.CSV_DIR_PATH) if f.endswith("csv")]
        now = datetime.datetime.now()
        conv = lambda x, y, z: datetime.datetime(year=int(x), month=int(y), day=int(z),
                                                 tzinfo=now.tzinfo)
        # get a list of dates for files.
        dts = [conv(
            *(os.path.basename(f) if "." not in os.path.basename(f) else os.path.basename(f).split(".")[0]).split('-'))
               for f in files]
        dts = [d for d in sorted(dts) if startdate <= d <= enddate]

        out = None
        for d in dts:
            r = f"{d.year}-{d.month}-{d.day}.csv"
            df = pd.read_csv(os.path.join(Params.CSV_DIR_PATH, r), parse_dates=False)
            if out is None:
                out = df
            else:
                out = pd.concat([out, df], axis=0, ignore_index=True)
        out: pd.DataFrame = out
        #log_line(f"Shape of Out: {out.shape}; \n\t{out.columns=}")
        # as there is no index (besides identity, or occurenceID, which are pseudo-random), we can just encode records directly.
        return out

    @staticmethod
    @app.route('/aladata', methods=['GET'])
    @app.route('/aladata/', methods=['GET'])
    @app.route('/aladata/<daterange>', methods=['GET'])
    @app.route('/aladata/<daterange>/', methods=['GET'])
    @log_it(LogLevel.L_DEBUG, print_results=False, loggerName="methods")
    def get_aladata(daterange: str = ""):
        if daterange == "":
            #print("Err1")
            msg = HSM.HTTPMessages.Bad_Request.value.set_params(
                reason="No daterange specified. Requires 'DDMMYYYY-DDMMYYYY' for 'startdate-enddate' or literal ['weekly', 'monthly', or 'quarterly']")
            return (msg.get(), msg.status)

        sd, ed = None, None
        if daterange == "weekly":
            sd, ed = get_week_boundaries(dt.now())
        elif daterange == "monthly":
            sd, ed = get_month_boundaries(dt.now())
        elif daterange == "quarterly":
            sd, ed = get_quarter_boundaries(dt.now())
        else:
            dates = daterange.split("-")
            if len(dates) != 2:
                #print("Err2")
                msg = HSM.HTTPMessages.Bad_Request.value.set_params(
                    reason="Incorrect daterange specification. Requires 'DDMMYYYY-DDMMYYYY' for 'startdate-enddate'.")
                return (msg.get(), msg.status)

            # two strings
            sd, ed = dates[0], dates[1]
            if len(sd) != 8 or len(ed) != 8:
                #print("Err3")
                msg = HSM.HTTPMessages.Bad_Request.value.set_params(
                    reason="Incorrect daterange specification. Requires 'DDMMYYYY-DDMMYYYY' for 'startdate-enddate'.")
                return msg.get(), msg.status
            try:
                sd = GetRecordsResource.handle_DDMMYYYY(sd)
                ed = GetRecordsResource.handle_DDMMYYYY(ed)
            except BaseException as BE:
                msg = HSM.HTTPMessages.Bad_Request.value.set_params(
                    reason="Incorrect daterange specification. Requires integers in 'DDMMYYYY-DDMMYYYY' for 'startdate-enddate'.")
                return msg.get(), msg.status

        # get the data to send back.
        dfjson = GetRecordsResource.get_data_df(sd, ed)
        # construct HTTP return message.
        if dfjson is not None:
            #print(f"{type(dfjson)=}, {dfjson[:1]=}")
            msg = HSM.HTTPMessages.Ok.value.set_params(dicta=dict(aladata=dfjson.to_dict(orient='records'), startdate=sd.isoformat(),
                                                                  enddate=ed.isoformat()))
        else:
            msg = HSM.HTTPMessages.Not_Found.value.set_params(reason="None Exist in Time Period",
                                                              startdate=sd.isoformat(),
                                                              enddate=ed.isoformat())
        return (msg.get(), msg.status)
    @staticmethod
    def get():
        # to appease swagger, so it doesn't blow us up with a ValueError
        pass


api.add_resource(GetRecordsResource, "/aladata", "/aladata/", "/aladata/<daterange>", "/aladata/<daterange>/")
api.add_resource(UploadResource, "/upload", "/upload/", "/upload/<target>", "/upload/<target>/")
api.add_resource(CheckClassificationResource, "/check_classify/<identity>", "/check_classify/<identity>/")

if __name__ == '__main__':
    print("This is Main")
    app.run()
    #serve(app, host='0.0.0.0', port=50010, threads=1)
    pass
