from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import time
import threading
import os
import requests
import json

# Initialization for MTCNN, InceptionResnetV1, dataset, and embeddings
mtcnn0 = MTCNN(image_size=240, margin=0, keep_all=False, min_face_size=40)
mtcnn = MTCNN(image_size=240, margin=0, keep_all=True, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
exit_program = False
dataset = datasets.ImageFolder('entry/media/photos')
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}


def collate_fn(x):
    return x[0]


loader = DataLoader(dataset, collate_fn=collate_fn)
loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = []
embedding_list = []

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob > 0.92:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])

# Save and load data for embeddings and names
data = [embedding_list, name_list]
torch.save(data, 'data.pt')
load_data = torch.load('data.pt')
embedding_list, name_list = load_data

min_distance_threshold = 1

person_records = {}
last_seen = {}
camera_records = {}  # Dictionary to hold lists of records for each camera


def send_to_backend(person_data):
    backend_url = "http://127.0.0.1:8000/camera-history/add/"
    # Note the change here: using the json parameter instead of data and removing the manual headers
    print(person_data)  # Ensure this print statement is within the function scope
    response = requests.post(backend_url, json=person_data)
    if response.status_code == 201 or response.status_code == 200:
        print("Data sent to the backend successfully.")
    else:
        print(
            f"Failed to send data to the backend. Status code: {response.status_code}")


def camera_feed_process(camera_index, exit_signal):
    global person_records, camera_records
    cam = cv2.VideoCapture(camera_index)
    # window_name = f"Camera {camera_index}"
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while not exit_signal.is_set():
        ret, frame = cam.read()
        if not ret:
            # print(f"Failed to grab frame from camera {camera_index}, try again")
            break

        img = Image.fromarray(frame)
        img_cropped_list, prob_list = mtcnn(img, return_prob=True)
        # cv2.imshow(window_name, frame)

        current_seen_names = []

        if img_cropped_list is not None:
            boxes, _ = mtcnn.detect(img)

            for i, (img_cropped, prob) in enumerate(zip(img_cropped_list, prob_list)):
                if prob > 0.9:
                    emb = resnet(img_cropped.unsqueeze(0)).detach()
                    dist_list = [torch.dist(emb, emb_db).item()
                                 for emb_db in embedding_list]
                    min_dist = min(dist_list)
                    if min_dist > min_distance_threshold:
                        name = "Unknown"
                    else:
                        min_dist_idx = dist_list.index(min_dist)
                        name = name_list[min_dist_idx]

                    current_seen_names.append(name)
                    box = boxes[i]
                    frame = cv2.rectangle(frame, (int(box[0]), int(
                        box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(frame, name, (int(box[0]), int(
                        box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    if name != "Unknown":
                        if name not in person_records or (name in person_records and person_records[name]['exit_time'] is not None):
                            entry_time = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                            person_records[name] = {
                                'entry_time': entry_time, 'exit_time': None, 'camera_index': camera_index}
                            camera_records.setdefault(camera_index, []).append(
                                (name, entry_time, None))
                        last_seen[name] = {'count': 0,
                                           'camera_index': camera_index}

        # Adjusted for camera index specific exit logic
        for name, info in list(last_seen.items()):
            if name not in current_seen_names and info['camera_index'] == camera_index:
                last_seen[name]['count'] += 1
                if last_seen[name]['count'] > 100:
                    if name in person_records and person_records[name]['exit_time'] is None and person_records[name]['camera_index'] == camera_index:
                        exit_time = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        person_records[name]['exit_time'] = exit_time
                        # Prepare data for sending to backend
                        name1, id1 = name.split("_")
                        id1 = int(id1)
                        person_data = {
                            'name': name1,
                            'id': id1,
                            'checkIn_time': person_records[name]['entry_time'],
                            'checkOut_time': person_records[name]['exit_time'],
                            'camera_id': camera_index
                        }
                        # Send data to the backend
                        send_to_backend(person_data)

                        # print(f'{name} entered at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry_time))} and exited at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exit_time))} from camera {camera_index}')
                        for record in camera_records[camera_index]:
                            # Find the matching record and update the exit time
                            if record[0] == name and record[2] is None:
                                camera_records[camera_index].append(
                                    (name, record[1], exit_time))
                                camera_records[camera_index].remove(record)
                                break
                    del last_seen[name]
            elif name in current_seen_names:
                last_seen[name] = {'count': 0, 'camera_index': camera_index}

        yield frame

    cam.release()


if __name__ == "__main__":
    exit_signal = threading.Event()
    available_indices = [index for index in range(
        5) if cv2.VideoCapture(index).isOpened()]

    threads = []
    for index in available_indices:
        thread = threading.Thread(
            target=camera_feed_process, args=(index, exit_signal))
        threads.append(thread)
        thread.start()

    while not exit_signal.is_set():
        time.sleep(0.1)  # Reduce CPU usage

    for thread in threads:
        thread.join()

    cv2.destroyAllWindows()  # Ensure all windows are closed here
    for camera_index in camera_records:
        print(f"\nRecords for Camera {camera_index}:")
        for record in camera_records[camera_index]:
            person, entry_time, exit_time = record
            entry_str = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(entry_time))
            exit_str = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(exit_time)) if exit_time else "Still inside"
            print(f"{person} entered at {entry_str} and exited at {exit_str}")
