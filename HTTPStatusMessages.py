from enum import Enum
from flask import jsonify


class Message:
    status = 100
    message = "N/A"

    def __init__(self, status, message, other: dict | None=None):
        """
        Used in the Enumeration HTTPMessages to construct a list of all HTTP messages (errors, statuses, requests etc..) in accordance with an abstracted version of the HTTP standard. (See also `https://developer.mozilla.org/en-US/docs/Web/HTTP/Messages`)

        Note that not all codes are functional or applicable to all situations. Please see the HTTP specifications for details. (https://developer.mozilla.org/en-US/docs/Web/HTTP/Resources_and_specifications)
        :param status: int representing the code of this HTTP Message
        :param message: str representing the simple human-readable message corresponding to Status.
        :param other: details to add (extenally - outside the ISO standards for additional information to the user - assigned at use)
        """
        self.status = status
        self.message = message
        self.other = other

    def set_params(self, dicta=None, **kwargs):
        """
        Delegate method to handle the assignment of `self.other`.
        :param dicta: as a dictionary to assign directly.
        :param kwargs: or, use keyword arguments to construct kwargs dict manually.
        :return:
        """
        if dicta is None:
            self.other = kwargs
        else:
            self.other = dicta
        return self

    def get(self):
        """
        Used to return a json-compatible version of this instance, adding in the additional parameters of `self.other` if it exists.
        :return:
        """
        if self.other is None:
            return jsonify(dict(status=self.status, message=self.message))
        else:
            odict = dict(status=self.status, message=self.message)
            odict.update(self.other)
            return jsonify(odict)

    def get_dict(self):
        """
        The same as `self.get` but not json-ified.
        :return:
        """
        if self.other is None:
            return jsonify(dict(status=self.status, message=self.message))
        else:
            odict = dict(status=self.status, message=self.message)
            odict.update(self.other)
        return odict

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if type(other) == Message:
            return other.status == self.status
        elif type(other) == int:
            return other == self.status
        else:
            return other == str(self.status)


class HTTPMessages(Enum):
    """
    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
    for a list of all HTTP codes, including ones that aren't implemented here.
    """
    # Information Responses
    Continue = Message(100, 'Continue')
    Processing = Message(102, 'Processing')
    # Successful Responses
    Ok = Message(200, 'OK')
    Created = Message(201, 'Created')
    Accepted = Message(202, 'Accepted')
    Non_Authoritative_Information = Message(203, 'Non-Authoritative Information')
    No_Content = Message(204, 'No Content')
    Reset_Content = Message(205, 'Reset Content')
    Partial_Content = Message(206, 'Partial Content')
    Multi_Status = Message(207, 'Multi Status')
    Already_Reported = Message(208, 'Already Reported')
    # Redirection Messages
    Multiple_Choices = Message(300, 'Multiple Choices')
    Moved_Permanently = Message(301, 'Moved Permanently')
    Found = Message(302, 'Found')
    See_Other = Message(303, 'See Other')
    Not_Modified = Message(304, 'Not Modified')
    Temporary_Redirect = Message(307, 'Temporary Redirect')
    Permanent_Redirect = Message(308, 'Permanent Redirect')
    # Client Error Responses
    Bad_Request = Message(400, 'Bad Request')
    Unauthorized = Message(401, 'Unauthorized')
    PaymentRequired = Message(402, 'Payment Required')
    Forbidden = Message(403, 'Forbidden')
    Not_Found = Message(404, 'Not Found')
    Method_Not_Allowed = Message(405, 'Method Not Allowed')
    Not_Acceptable = Message(406, 'Not Acceptable')
    Proxy_Authentication_Required = Message(407, 'Proxy Authentication Required')
    Request_Timeout = Message(408, 'Request Timeout')
    Conflict = Message(409, 'Conflict')
    Gone = Message(410, 'Gone')
    LengthRequired = Message(411, 'Length Required')
    Precondition_Failed = Message(412, 'Precondition Failed')
    Payload_Too_Large = Message(413, 'Payload Too Large')
    URI_Too_Long = Message(414, 'URI Too Long')
    Unsupported_Media_Type = Message(415, 'Unsupported Media Type')
    Range_Not_Satisfiable = Message(416, 'Range Not Satisfiable')
    Expectation_Failed = Message(417, 'Expectation Failed')
    Im_a_Teapot = Message(418, "I'm a Teapot (short and stout).")
    Misdirected_Request = Message(421, "Mis-directed Request")
    Unprocessable_Content = Message(422, "Unprocessable Content")
    Locked = Message(423, "Locked")
    Failed_Dependency = Message(424, "Failed Dependency")
    Too_Early = Message(425, "Too Early")
    Upgrade_Required = Message(426, "Upgrade Required")
    Precondition_Required = Message(428, "Precondition Required")
    Too_Many_Requests = Message(429, 'Too Many Requests')
    Request_Header_Fields_Too_Large = Message(431, 'Request Header Fields Too Large')
    Unavailable_For_Legal_Reasons = Message(451, "Unavailable For Legal Reasons")

    # Server Error Responses
    Internal_Server_Error = Message(500, "Internal Server Error")
    Not_Implemented = Message(501, "Not Implemented")
    Bad_Gateway = Message(502, "Bad Gateway")
    Service_Unavailable = Message(503, 'Service Unavailable')
    Gateway_Timeout = Message(504, 'Gateway Timeout')
    HTTP_Version_Not_Supported = Message(505, "HTTP Version Not Supported")
    Variant_Also_Negotiates = Message(506, 'Variant also Negotiates')
    Insufficient_Storage = Message(507, "Insufficient Storage")
    Loop_Detected = Message(508, 'Loop Detected')
    Not_Extended = Message(510, 'Not Extended')
    Network_Authentication_Required = Message(511, 'Network Authentication Required')




