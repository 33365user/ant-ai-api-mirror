import os
from loggermod import LogLevel, log_it
from datetime import datetime, timedelta

class Params:
    UPLOAD_BASE_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), "uploads"))
    ALLOWED_UPLOAD_EXTENSIONS = {"jpg", "jpeg", "tif", "tiff"}
    UPLOAD_API_VERSION = 1000
    MAX_UPLOAD_FILESIZE = 128 * 1024 * 1024
    API_VERSION = "0.1.0.0"
    API_QUERY_AWAIT = timedelta(seconds=5) # time between which each individual on-going request may request an update from the AI about it's request.
    MAX_AI_QUEUE_SIZE = 500  # 500 requests for classification at any one time
    QUEUE_TIMEOUT = timedelta(minutes=12)  # if a request has been in the queue for more than 10 minutes, drop it.
    # once images have been uploaded and classified, the images will be moved to the IMAGE_STORAGE_SAVE_PATH for safe-
    # keeping if SAVE_IMAGES_AFTER_CLASSIFICATION is True, else it will be deleted.
    IMAGE_STORAGE_SAVE_PATH = os.path.abspath(os.path.join(os.getcwd(), "images"))
    SAVE_IMAGES_AFTER_CLASSIFICATION = True


    DB_LOG_DIR_PATH = os.path.normpath("./dblog")
    DB_LOG_LEVEL: LogLevel = LogLevel.L_INFO
    DB_LOG_FORMAT = "%(process)d : %(name)s | %(levelname)s : %(asctime)s :: %(msg)s "

    CSV_DIR_PATH = os.path.normpath("./csvs")
    # perhaps generate a unique hash ID instead for occurenceID
    # parameters required: occurenceID, basisOfRecord="HumanObservation"
    # geodeticDatum (https://dwc.tdwg.org/terms/#dwc:geodeticDatum)
    # eventDate (https://dwc.tdwg.org/terms/#dwc:eventDate)
    # scientificName
    CSV_COLUMNS = ["occurenceID", "basisOfRecord", "decimalLatitude", "decimalLongitude", "geodeticDatum",
                   "coordinateUncertaintyInMeters", "eventDate", "uri", "identity", "scientificName", "score", "regionsOfInterest", "instanceN"]
    # nb: if multiple ants are detected, we need to make multiple records in the csv
    CSV_MAKE_NEW_DELTA = timedelta(weeks=2)

    @classmethod
    def allow_version(cls, remote):
        return remote >= cls.UPLOAD_API_VERSION

    @classmethod
    def allowed_file(cls, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in cls.ALLOWED_UPLOAD_EXTENSIONS
