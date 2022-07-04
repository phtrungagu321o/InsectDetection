import json
import cv2
from django.http import StreamingHttpResponse
from django.http import HttpResponse, Http404
from django.http import HttpResponseRedirect
from rich import print
from django.shortcuts import render
from yolov5.utils.general import (check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.utils.plots import Annotator, colors
import torch
from yolov5.utils.torch_utils import select_device
from django.core.files.storage import default_storage
from norfair import Video
import time
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s_best.pt')


# model = yolov5.load('yolov5s.pt')


def home(request):
    return render(request, 'home.html')


def home_img(request):
    return render(request, 'home_img.html')


def home_results_img(request):
    if request.method == 'POST':
        model.conf = 0.65
        file = request.FILES["imageFile"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        img = file_url
        # im = ImageGrab.grab()
        results = model(img, size=640)
        result = results.pandas().xyxy[0].to_json(orient="records")
        parsed = json.loads(result)
        try:
            label = parsed[0]['name']
            path_save = str('static/home/img/exp/' + label)
            results.save(save_dir=path_save)
            results.crop(save_dir=path_save)
            path_save1 = str('home/img/exp/' + label + '/' + file_name)
            path_crop = str('home/img/exp/' + label + '/crops/' + label + '/' + file_name)
            message = 'Nhận dạng thành công loài '
            return render(request, "home_img.html",
                          {"name_path": path_save1, "label": label, 'message': message, 'path_crop': path_crop})

        except:
            message = 'không thể nhận dạng được'
            label = ''
            path_save1 = str('home/img/default.png')
            return render(request, "home_img.html", {"name_path": path_save1, 'message': message, 'label': label})
    return HttpResponseRedirect(request.META['HTTP_REFERER'])


max_distance_between_points: int = 30

device = select_device('0')
# initialize deepsort
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort('osnet_x0_25',
                    device,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    )

# Get names and colors
names = model.module.names if hasattr(model, 'module') else model.names


def home_video(request):
    return render(request, 'home_video.html')


def home_results_video(request):
    if request.method == 'POST':
        model.conf = 0.65
        file = request.FILES["imageFile_video"]
        file_name = default_storage.save(file.name, file)
        file_url = default_storage.path(file_name)
        video_path = file_url
        video_capture = cv2.VideoCapture(video_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0
        start = time.time()
        # path_video
        video = Video(input_path=video_path)
        output_path = str('.')
        video_path_complete = ''
        output_path_is_dir = os.path.isdir(output_path)
        if output_path_is_dir and video_path is not None:
            base_file_name = video_path.split("/")[-1].split(".")[0]
            print(base_file_name)
            file_name = base_file_name + "_out.mp4"
            video_path_complete = os.path.join(output_path, file_name)
            print(video_path_complete)
        else:
            video_path_complete = output_path
            print(video_path_complete)
        list_name_conf = [
        ]
        list_name = [

        ]
        print()
        for frame in video:
            frame_counter += 1
            process_fps = str(frame_counter / (time.time() - start))
            print("FPS: " + process_fps)
            results = model(frame, augment=True)
            # proccess
            annotator = Annotator(frame, line_width=2, pil=not ascii)
            det = results.pred[0]
            if det is not None and len(det):
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        # print(str('tên class: ' + names[c] + ' - độ chính xác: ' + conf))
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        if names[c] not in list_name:
                            list_name.insert(0, names[c])
                            list_name_conf.insert(0, {'name': names[c], 'conf': float(conf)})
                        else:
                            for i in range(0, len(list_name_conf)):
                                if list_name_conf[i]['name'] == names[c] and list_name_conf[i]['conf'] < float(conf):
                                    list_name_conf[i]['conf'] = float(conf)
                                    break
                        print(list_name)
                        print(list_name_conf)
            else:
                deepsort.increment_ages()

            video.write(frame)
            # yield time_path
        return render(request, 'home_video.html',
                      {'name_video': video_path_complete, 'list_name': list_name, 'list_name_conf': list_name_conf})
    return render(request, 'home_video.html')


def download_video(request, path):
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(path)
            return response
    raise Http404


def webcam(request):
    return render(request, 'home_webcam.html')


def stream():
    cap = cv2.VideoCapture(0)
    model.conf = 0.65
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break

        results = model(frame, augment=True)
        # proccess
        annotator = Annotator(frame, line_width=2, pil=not ascii)
        det = results.pred[0]
        if det is not None and len(det):
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):
                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    # print(str('tên class: ' + names[c] + ' - độ chính xác: ' + conf))
                    annotator.box_label(bboxes, label, color=colors(c, True))
                    print("****")
                    print(names[c])
                    print("----")
        else:
            deepsort.increment_ages()
        im0 = annotator.result()
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')
