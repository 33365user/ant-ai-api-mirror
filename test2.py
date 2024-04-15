import urllib as ul
import requests
import time
import json
# this module attempts to assert that the app.py is functional according to testing criteria.


def convert(s: bytes):
    return json.loads(s.decode('utf-8'))


files = {"file": open("testimg.jpg", 'rb')}
headers = {'User-Agent': 'insomnia/8.2.0'}  # seems that headers are required - which seems to be applicable.

# Attempt to send the file to the API - we are testing for functionality.
urlbase = "http://127.0.0.1:5000/"
response = requests.post(urlbase + "upload/classify", files=files, headers=headers)
rdict: dict = convert(response.content)
# should report
assert type(rdict) == dict
assert rdict.get("identity", None) is not None
assert rdict.get("status", None) == 201

identity = rdict.get("identity")

# check classify
respcode = 0

while respcode != 200:
    response = requests.get(urlbase + f"check_classify/{identity}", headers=headers)
    rdict: dict = convert(response.content)
    assert type(rdict) == dict
    assert rdict.get("status", None) is not None
    respcode = rdict.get("status")
    print(f"Current Respcode: {respcode}")
    time.sleep(2)

print("Succeessful!")
assert rdict.get("status", None) == 200
print(rdict)

# try and check classification again - should fail
response = requests.get(urlbase + f"check_classify/{identity}", headers=headers)
rdict: dict = convert(response.content)
assert type(rdict) == dict
assert rdict.get("status") == 404
print("Failed Successfully!")
print(rdict)
