from flask import Flask, Response, jsonify
from flask_cors import CORS
import threading
import cv2
import os  # Import os to read environment variables

# Import camera_feed_process which handles recognition
from main import camera_feed_process

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

cameras = {}
exit_signals = {}


def start_cameras():
    """Initialize cameras and allocate resources."""
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            break
        exit_signals[index] = threading.Event()
        # Note: Each camera is now associated with its own thread for processing.
        cameras[index] = threading.Thread(
            target=camera_feed_process, args=(index, exit_signals[index]))
        cameras[index].start()
        index += 1


def stop_cameras():
    """Release all cameras and set exit signals."""
    for index in cameras.keys():
        exit_signals[index].set()
        cameras[index].join()


def generate_frames(camera_index):
    """Yield frames from camera_feed_process, managed by main.py."""
    cap = cv2.VideoCapture(camera_index)
    exit_signal = exit_signals[camera_index]
    if cap is None or not cap.isOpened():
        return  # Early exit if the camera is not available

    for frame in camera_feed_process(camera_index, exit_signal):
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/video_feed/<int:camera_index>', methods=['GET'])
def video_feed(camera_index):
    """Stream video feed for each camera with facial recognition."""
    return Response(generate_frames(camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """API endpoint to list all active cameras."""
    return jsonify(list(cameras.keys()))


if __name__ == "__main__":
    try:
        start_cameras()
        # port = int(os.environ.get('PORT', 5000))
        # app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
        app.run(debug=False)
    finally:
        stop_cameras()
