#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response, request, make_response
import happybase

# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera


# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)


@app.route('/updateCar', methods=['POST', 'GET'])
def updateCar():
    print("Success updateCar")
    car_data = request.get_json(force=True)
    row_key = car_data['row_key']
    plate_num = car_data['changeNum']
    print('row_key :' + str(row_key) + 'plate_num' + str(car_data))
    connection = happybase.Connection('server01.hadoop.com')
    connection.open()
    table = connection.table('car_data')
    table.put(row_key, {'category:plate_num': plate_num})
    # table.put('uuid내용을 적음', {'컬럼명' : '입력할 밸류값'})
    response = make_response("X")

    return response


@app.route('/streaming/<string:video_id>')
def index(video_id):
    """Video streaming home page."""
    print("check def index")
    path = request.args.get("path")
    os.environ['MEDIA'] = path.replace('"', "")+video_id
    print("path :", os.environ['MEDIA'])
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        print("check def gen")
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    print("check def video_feed")
    return Response(gen(Camera()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
