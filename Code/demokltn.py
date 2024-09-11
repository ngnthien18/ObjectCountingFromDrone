import numpy as np
import cv2
import io
import os
import shutil
import glob
import torch
from collections import deque
from random import randrange
from ultralytics import YOLO
from sort import Sort
from SFSORT import SFSORT
import PySimpleGUI as sg

def initialize_model(model_path):
    return YOLO(model_path)

def initialize_tracker(tracker_type,tracker_params):
    if tracker_type == "SORT":
        return Sort(max_age=tracker_params['max_age'],
                    min_hits=tracker_params['min_hits'],
                    iou_threshold=tracker_params['iou_threshold'])
    elif tracker_type == "SFSORT":
        return SFSORT(tracker_params)

def process_frame(frame, model, trackers, classes, roi, colors, histories, counts, tracker_type):
    results = model(frame)[0]
    detections = {class_name: np.empty((0, 5)) for class_name in classes}

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_name = classes[int(class_id)]
        detections[class_name] = np.vstack([detections[class_name], [x1, y1, x2, y2, score]])

    for class_name, tracker in trackers.items():
        if tracker_type == "SORT":
            tracks = tracker.update(detections[class_name])
        elif tracker_type == "SFSORT":
            tracks = tracker.update(detections[class_name][:, :4], detections[class_name][:, -1])
        visualize_tracks(frame, tracks, class_name, colors, histories[class_name], counts[class_name], roi, tracker_type)

def visualize_tracks(frame, tracks, class_name, colors, history, count_set, roi, tracker_type):
    if len(tracks) == 0:
        return
    
    if tracker_type == "SORT":
        for d in tracks:
            x1, y1, x2, y2, track_id = d
            track_id = int(track_id)
            if track_id not in colors:
                colors[track_id] = (randrange(255), randrange(255), randrange(255))
            color = colors[track_id]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3, cv2.LINE_AA)
            cv2.putText(frame, f'{track_id} {class_name}', (int(x1), int(y1 - 4)), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)
            flag = cv2.pointPolygonTest(np.array(roi, dtype=np.int32), (int((x1 + x2) / 2), int((y1 + y2) / 2)), False)
            if flag > 0:
                count_set.add(track_id)
            if track_id not in history:
                history[track_id] = deque(maxlen=25)
            history[track_id].appendleft((int((x1 + x2) / 2), int((y1 + y2) / 2)))
            for i in range(1, len(history[track_id])):
                cv2.line(frame, history[track_id][i - 1], history[track_id][i], color, 3)
    elif tracker_type == "SFSORT":
        bbox_list = tracks[:, 0]
        track_id_list = tracks[:, 1]
        for idx, (track_id, bbox) in enumerate(zip(track_id_list, bbox_list)):
            if track_id not in colors:
                colors[track_id] = (randrange(255), randrange(255), randrange(255))
            color = colors[track_id]
            x0, y0, x1, y1 = map(int, bbox)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
            cv2.putText(frame, f'{track_id} {class_name}', (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            flag = cv2.pointPolygonTest(np.array(roi, dtype=np.int32), (int((x0 + x1) / 2), int((y0 + y1) / 2)), False)
            if flag > 0:
                count_set.add(track_id)
            if track_id not in history:
                history[track_id] = deque(maxlen=25)
            history[track_id].appendleft((int((x0 + x1) / 2), int((y0 + y1) / 2)))
            for i in range(1, len(history[track_id])):
                cv2.line(frame, history[track_id][i - 1], history[track_id][i], color, 2)

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

def convert_to_bytes(frame):
    is_success, buffer = cv2.imencode(".png", frame)
    io_buf = io.BytesIO(buffer)
    return io_buf.getvalue()

def resize_frame(frame, width=None, height=None):
    (h, w) = frame.shape[:2]
    return cv2.resize(frame, (width, height)), (w, h)

def create_yaml(project_name, train_data_path, class_names, save_directory):
    train_path = os.path.join(train_data_path, 'train').replace('\\', '/')
    val_path = os.path.join(train_data_path, 'val').replace('\\', '/')

    yaml_content = f"""train: {train_path}
val: {val_path}
nc: {len(class_names)}
names: [{', '.join(f"'{name}'" for name in class_names)}]
"""
    print(f"Project Name: {project_name}")
    yaml_path = os.path.join(save_directory, f'{project_name}.yaml').replace('\\', '/')
    print(f"YAML Path: {yaml_path}")
    with open(yaml_path, 'w') as file:
        file.write(yaml_content)
    return yaml_path

def copy_and_remove_latest_run_files(model_save_path, project_name):
    list_of_dirs = glob.glob('C:/Users/Dark Angel/runs/detect/' + project_name)
    if not list_of_dirs:
        print("No 'C:/Users/Dark Angel/runs/detect/" + project_name + "' directories found. Skipping copy and removal.")
        return

    latest_dir = max(list_of_dirs, key=os.path.getmtime)

    if os.path.exists(latest_dir):
        for item in os.listdir(latest_dir):
            s = os.path.join(latest_dir, item)
            d = os.path.join(model_save_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    runs_dir = 'C:/Users/Dark Angel/runs'
    if os.path.exists(runs_dir) and os.path.isdir(runs_dir):
        shutil.rmtree(runs_dir)

def train_yolo(data_yaml, model_type, img_size, epochs, model_save_path, project_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(f'{model_type}.pt').to(device)
    results = model.train(data=data_yaml, epochs=epochs, imgsz=img_size, name=project_name, save=True)
    copy_and_remove_latest_run_files(model_save_path, project_name)
    return results

def main_counting(model_path, input_video, tracker_type, tracker_params=None):
    global roi
    model = initialize_model(model_path)
    class_names = list(model.names.values())
    cap = cv2.VideoCapture(input_video)

    trackers = {class_name: initialize_tracker(tracker_type, tracker_params) for class_name in class_names}
    colors = {}
    histories = {class_name: {} for class_name in class_names}
    counts = {class_name: set() for class_name in class_names}

    sg.theme('LightBlue2')
    layout = [[sg.Image(filename='', key='image')], [sg.Button('Exit')]]
    window = sg.Window('YOLO Object Counting', layout)
    prev_time = cv2.getTickCount()
    while True:
        event, values = window.read(timeout=0.000001)
        if event == sg.WIN_CLOSED or event == 'Exit':
            break

        ret, frame = cap.read()
        if not ret:
            break
        process_frame(frame, model, trackers, class_names, roi, colors, histories, counts, tracker_type)
        cv2.polylines(frame, [np.array(roi, dtype=np.int32)], True, (0, 255, 255), 3)

        y_offset = 25
        for class_name in class_names:
            cv2.putText(frame, f'{class_name}: {len(counts[class_name])}', (25, y_offset), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 45
        total_count = sum(len(counts[class_name]) for class_name in class_names)
        cv2.putText(frame, f'Total: {total_count}', (25, y_offset), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)
        current_time = cv2.getTickCount()
        time_elapsed = (current_time - prev_time) / cv2.getTickFrequency()
        prev_time = current_time
        fps = 1.0 / time_elapsed
        cv2.putText(frame, f'FPS: {fps:.2f}', (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) - 200, 25), 0, 1, (255, 255, 255), 2, cv2.LINE_AA)

        imgbytes = cv2.imencode('.png', frame)[1].tobytes()
        window['image'].update(data=imgbytes)

    cap.release()
    window.close()

if __name__ == '__main__':
    sg.theme('LightBlue2')
    model_sizes_v8 = [
        ("Nano", "n"),
        ("Small", "s"),
        ("Medium", "m"),
        ("Large", "l"),
        ("Xtra Large", "x")
    ]
    image = cv2.imread('drone.png')
    img = convert_to_bytes(image)
    sort_layout = [
        [sg.Column([
            [sg.Text('Model Path', size=(10, 1)), sg.InputText(default_text='D:/KLTN/demo2/best.pt', key='model_path'), sg.FileBrowse()],
            [sg.Text('Input Video', size=(10, 1)), sg.InputText(default_text='D:/KLTN/demo2/demo1.mp4', key='input_video'), sg.FileBrowse()],
            [sg.Text('SORT Parameters', key='sort_params_label')],
            [
                sg.Text('max_age', size=(7, 1)), sg.InputText(default_text='1', key='max_age', size=(5, 1)),
                sg.Text('min_hits', size=(7, 1)), sg.InputText(default_text='3', key='min_hits', size=(5, 1)),
                sg.Text('iou_threshold', size=(10, 1)), sg.InputText(default_text='0.3', key='iou_threshold', size=(5, 1))
            ],
            [sg.Button('Select ROI', key='roi')],
            [sg.Button('Run', key='run_sort'), sg.Button('Cancel')]]),
        sg.Column([
            [sg.Graph(
                canvas_size=(528, 297),
                graph_bottom_left=(0, 297),
                graph_top_right=(528, 0),
                key="-GRAPH-",
                enable_events=True,
                drag_submits=True)]
                ])
            ]
        ]

    sfsort_layout = [
        [sg.Column([
            [sg.Text('Model Path', size=(10, 1)), sg.InputText(default_text='D:/KLTN/demo2/best.pt', key='model_path_2'), sg.FileBrowse()],
            [sg.Text('Input Video', size=(10, 1)), sg.InputText(default_text='D:/KLTN/demo2/demo1.mp4',key='input_video_2'), sg.FileBrowse()],
            [sg.Text('SFSORT Parameters', key='sf_params_label')],
            [sg.Checkbox('Dynamic Tuning', default=True, key='dynamic_tuning')],
            [
                sg.Text('CTH', size=(15, 1)), sg.InputText(default_text='0.5', key='cth', size=(5, 1)),
                sg.Text('High TH', size=(15, 1)), sg.InputText(default_text='0.7', key='high_th', size=(5, 1)),
                sg.Text('High TH M', size=(15, 1)), sg.InputText(default_text='0.1', key='high_th_m', size=(5, 1)),
            ],
            [
                sg.Text('Match TH First', size=(15, 1)), sg.InputText(default_text='0.5', key='match_th_first', size=(5, 1)),
                sg.Text('Match TH First M', size=(15, 1)), sg.InputText(default_text='0.05', key='match_th_first_m', size=(5, 1)),
                sg.Text('Match TH Second', size=(15, 1)), sg.InputText(default_text='0.3', key='match_th_second', size=(5, 1)),
            ],
            [
                sg.Text('Low TH', size=(15, 1)), sg.InputText(default_text='0.2', key='low_th', size=(5, 1)),
                sg.Text('New Track TH', size=(15, 1)), sg.InputText(default_text='0.5', key='new_track_th', size=(5, 1)),
                sg.Text('New Track TH M', size=(15, 1)), sg.InputText(default_text='0.1', key='new_track_th_m', size=(5, 1)),
            ],
            [
                sg.Text('Marginal Timeout', size=(15, 1)), sg.InputText(default_text='0.7', key='marginal_timeout', size=(5, 1)),
                sg.Text('Central Timeout', size=(15, 1)), sg.InputText(default_text='1', key='central_timeout', size=(5, 1)),
                sg.Text('Horizontal Margin', size=(15, 1)), sg.InputText(default_text='0.1', key='horizontal_margin', size=(5, 1)),
            ],
            [
                sg.Text('Vertical margin', size=(15, 1)), sg.InputText(default_text='0.1', key='vertical_margin', size=(5, 1)),
                sg.Text('Frame Width', size=(15, 1)), sg.InputText(default_text='1', key='frame_width', size=(5, 1)),
                sg.Text('Frame Height', size=(15, 1)), sg.InputText(default_text='1', key='frame_height', size=(5, 1)),
            ],
            [sg.Button('Select ROI', key='roi2')],
            [sg.Button('Run', key='run_sfsort'), sg.Button('Cancel', key='Cancel2')]]),
        sg.Column([
            [sg.Graph(
                canvas_size=(528, 297),
                graph_bottom_left=(0, 297),
                graph_top_right=(528, 0),
                key="-GRAPH2-",
                enable_events=True,
                drag_submits=True)]
                ])
            ]
        ]
    train_layout = [
        [sg.Column([
            [sg.Text('Project Name', size=(10, 1)), sg.InputText(default_text='drone', key='project_name')],
            [sg.Text('Train Data', size=(10, 1)), sg.InputText(default_text='D:/KLTN/data', key='train_data'), sg.FolderBrowse()],
            [sg.Text('Save Folder', size=(10, 1)), sg.InputText(default_text='D:/KLTN/train_result', key='save_folder'), sg.FolderBrowse()],
            [sg.Text('Image Size', size=(10, 1)), sg.InputText(default_text='640', key='imgsz')],
            [sg.Text('Epochs', size=(10, 1)), sg.InputText(default_text='50', key='epochs')],
            [sg.Text('Class Names', size=(10, 1)), sg.InputText(default_text='camel', key='class_names_3')],
            [sg.Text('Choose verson of YOLOv8', key='yolov8')],
            [sg.Radio(text, "MODEL_SIZE", key=f"RADIO_{value}") for text, value in model_sizes_v8],
            [sg.Button('Train', key='train'), sg.Button('Cancel')]]),
        sg.Column([
            [sg.Image(data=img)]
        ])
    ]]
    layout = [
        [sg.TabGroup([
            [sg.Tab('SORT', sort_layout)],
            [sg.Tab('SFSORT', sfsort_layout)],
            [sg.Tab('YOLO Drone Training', train_layout)]
        ])]
    ]
    window = sg.Window('Object Counting System From UAVs', layout)

    drawing = False
    start_point = None
    rect_id = None
    graph = window['-GRAPH-']
    graph2 = window['-GRAPH2-']
    original_size = None
    roi = None
    resized_dims = (528, 297)

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel' or event == 'Cancel2':
            break

        if event == 'roi' or event == 'roi2':
            input_video = values['input_video'] if event == 'roi' else values['input_video_2']
            frame = get_first_frame(input_video)
            if frame is not None:
                resized_frame, original_size = resize_frame(frame, width=resized_dims[0], height=resized_dims[1])
                img_bytes = convert_to_bytes(resized_frame)
                if event == 'roi':
                    graph.erase()
                    graph.draw_image(data=img_bytes, location=(0, 0))
                else:
                    graph2.erase()
                    graph2.draw_image(data=img_bytes, location=(0, 0))
            else:
                sg.popup('Could not load frame from video.')
            
        if event == "-GRAPH-" or event == "-GRAPH2-" and graph is not None:
            x, y = values['-GRAPH-'] if event == '-GRAPH-' else values['-GRAPH2-']
            if not drawing:
                start_point = (x, y)
                drawing = True
            else:
                if event =='-GRAPH-':
                    if rect_id is not None:
                        graph.delete_figure(rect_id)
                    
                    end_point = (x, y)
                    rect_id = graph.draw_rectangle(start_point, end_point, line_color='yellow', line_width=3)
                if event =='-GRAPH2-':
                    if rect_id is not None:
                        graph2.delete_figure(rect_id)
                    
                    end_point = (x, y)
                    rect_id = graph2.draw_rectangle(start_point, end_point, line_color='yellow', line_width=3)
        elif event == "-GRAPH-+UP" or event == "-GRAPH2-+UP":
            drawing = False
            x1, y1 = start_point
            x2, y2 = end_point
            
            x2 = min(max(x2, 0), resized_dims[0])
            y2 = min(max(y2, 0), resized_dims[1])

            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)

            if original_size:
                orig_w, orig_h = original_size
                ratio_w = orig_w / resized_dims[0]
                ratio_h = orig_h / resized_dims[1]

                orig_x1 = int(left * ratio_w)
                orig_y1 = int(top * ratio_h)
                orig_x2 = int(right * ratio_w)
                orig_y2 = int(bottom * ratio_h)

                roi = [(orig_x1, orig_y1), (orig_x2, orig_y1), (orig_x2, orig_y2), (orig_x1, orig_y2)]
        if event == 'run_sort':
            tracker_type = 'SORT'
            model_path = values['model_path']
            input_video = values['input_video']
            tracker_params = {
                "max_age": int(values['max_age']), "min_hits": int(values['min_hits']), "iou_threshold": float(values['iou_threshold'])
            }
            if roi is not None:
                main_counting(model_path, input_video, tracker_type, tracker_params)
            else:
                sg.popup('Please select ROI.')
        if event == 'run_sfsort':
            tracker_type = 'SFSORT'
            model_path = values['model_path_2']
            input_video = values['input_video_2']
            cap = cv2.VideoCapture(input_video)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            tracker_params = {
                "dynamic_tuning": values['dynamic_tuning'], "cth": float(values['cth']),
                "high_th": float(values['high_th']), "high_th_m": float(values['high_th_m']),
                "match_th_first": float(values['match_th_first']), "match_th_first_m": float(values['match_th_first_m']),
                "match_th_second": float(values['match_th_second']), "low_th": float(values['low_th']),
                "new_track_th": float(values['new_track_th']), "new_track_th_m": float(values['new_track_th_m']),
                "marginal_timeout": float(values['marginal_timeout']) * frame_rate,
                "central_timeout": float(values['central_timeout']) * frame_rate,
                "horizontal_margin": float(values['horizontal_margin']) * frame_width,
                "vertical_margin": float(values['vertical_margin']) * frame_height,
                "frame_width": float(values['frame_width']) * frame_width,
                "frame_height": float(values['frame_height']) * frame_height
            }
            if roi is not None:
                main_counting(model_path, input_video, tracker_type, tracker_params)
            else:
                sg.popup('Please select ROI.')
        if event == 'train':
            project_name = values['project_name']
            train_data_path = values['train_data']
            class_names = values['class_names_3'].split(',')
            save_directory = values['save_folder']
            img_size = values['imgsz']
            epochs = int(values['epochs'])
            for text, value in model_sizes_v8:
                if values.get(f"RADIO_{value}"):
                    model_type = f"yolov8{value}"
            create_yaml(project_name,train_data_path,class_names,save_directory)
            data_yaml = os.path.join(save_directory, f'{project_name}.yaml')
            train_yolo(data_yaml, model_type, img_size, epochs, save_directory, project_name)
            print(f"Training completed. Model saved to {save_directory}")
    window.close()