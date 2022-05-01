import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

# leo 추가
from centroidtracker import CentroidTracker
import dlib
import imutils  # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)

import json
import websocket

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# table 상태 정보 저장 변수
jsonTable = {
    "table1": {
        "chair": {
            "up": 0,
            "down": 0
        },
        "object": {
            "notebook": 0,
            "book": 0,
            "bag": 0,
            "cup": 0
        }
    },

    "table2": {
        "chair": {
            "up": 0,
            "down": 0
        },
        "object": {
            "notebook": 0,
            "book": 0,
            "bag": 0,
            "cup": 0
        }
    },

    "table3": {
        "chair": {
            "up": 0,
            "down": 0
        },
        "object": {
            "notebook": 0,
            "book": 0,
            "bag": 0,
            "cup": 0
        }
    }
}

# table 정보 표시 판
tableBoard = [(0, 0), (530, 100)]

# table 자리에 앉은 객체 정보 저장
table1UpInfo = []
table1DownInfo = []
table2UpInfo = []
table2DownInfo = []
table3UpInfo = []
table3DownInfo = []

# table 위치 정보
table1 = [(20, 280), (270, 560)]
table2 = [(280, 280), (590, 560)]
table3 = [(610, 280), (930, 560)]

# cv2.putText 옵션
fontColor = (0, 255, 0)
fontThickness = 2

# 중심 추적 변수
ct1 = CentroidTracker(maxDisappeared=40, maxDistance=50)
ct2 = CentroidTracker(maxDisappeared=40, maxDistance=50)
ct3 = CentroidTracker(maxDisappeared=40, maxDistance=50)

# 추적 객체 목록
trackers1 = []
trackers2 = []
trackers3 = []

# 총 프레임 수
totalFrames = 0

# 프레임 크기 초기화(비디오에서 첫 번째 프레임을 읽는 즉시 설정)
W = None
H = None

# 마스크 정보
mask = 0
nomask = 0
error = 0

# 웹소켓
ws = websocket.WebSocket()
ws.connect("ws://35.77.144.191/ws/detectData")

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    # leo 추가
    global totalFrames, H, W
    # leo 추가 끝
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    videos = source.lower().endswith(('mp4', 'avi'))
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    elif videos:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # leo 추가 끝

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            # leo 추가
            im0 = imutils.resize(im0, width=1000)
            rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)  # RGB 변환

            if W is None or H is None:
                (H, W) = im0.shape[:2]

            # 객체 bounding box 목록
            rects1 = []
            rects2 = []
            rects3 = []

            # leo 추가 끝

            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):  # 프레임에서 탐지된 객체 수(det은 객체별 정보0~3 위치, 4는 conf, 5은 객체 번호(coco데이터 내 번호))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # 삭제 예정
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # 객체를 탐지할 때
            if totalFrames % 30 == 0:
                # 객체 추적 목록 초기화
                trackers1 = []
                trackers2 = []
                trackers3 = []

                boxes = []
                boxesXY = []
                confidences = []
                classIDs1 = []
                classIDs2 = []
                classIDs3 = []

                # layerOutputs 반복
                for detection in det:  # detection에는 객체 하나의 정보(det)
                    # 인식된 객체의 클래스 ID 및 확률 추출
                    classID = int(detection[5].tolist())  # detection에서 앞에 5개 값빼고 추출
                    confidence = float(detection[4].tolist())
                    # 마스크, 노마스크, error인 경우
                    if classID == 0:  # yolo-coco 디렉터리에 coco.names 파일을 참고하여 다른 object 도 인식 가능(0 인 경우 사람)
                        # 객체 확률이 최소 확률보다 큰 경우
                        if confidence > 0.25:
                            # bounding box  좌표
                            startX = int(detection[0:4][0].tolist())
                            startY = int(detection[0:4][1].tolist())
                            endX = int(detection[0:4][2].tolist())
                            endY = int(detection[0:4][3].tolist())

                            boxes.append([startX, startY, int(endX - startX), int(endY - startY)])
                            boxesXY.append([endX, endY])
                            confidences.append(float(confidence))

                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            tracker.start_track(rgb, rect)

                            classIDs1.append([classID, tracker])

                            # 인식된 객체를 추적 목록에 추가
                            trackers1.append(tracker)

                    elif classID == 1:  # yolo-coco 디렉터리에 coco.names 파일을 참고하여 다른 object 도 인식 가능(0 인 경우 사람)
                        # 객체 확률이 최소 확률보다 큰 경우
                        if confidence > 0.25:
                            # bounding box  좌표
                            startX = int(detection[0:4][0].tolist())
                            startY = int(detection[0:4][1].tolist())
                            endX = int(detection[0:4][2].tolist())
                            endY = int(detection[0:4][3].tolist())

                            boxes.append([startX, startY, int(endX - startX), int(endY - startY)])
                            boxesXY.append([endX, endY])
                            confidences.append(float(confidence))

                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            tracker.start_track(rgb, rect)

                            classIDs2.append([classID, tracker])

                            # 인식된 객체를 추적 목록에 추가
                            trackers2.append(tracker)

                    elif classID == 2:  # yolo-coco 디렉터리에 coco.names 파일을 참고하여 다른 object 도 인식 가능(0 인 경우 사람)
                        # 객체 확률이 최소 확률보다 큰 경우
                        if confidence > 0.25:
                            # bounding box  좌표
                            startX = int(detection[0:4][0].tolist())
                            startY = int(detection[0:4][1].tolist())
                            endX = int(detection[0:4][2].tolist())
                            endY = int(detection[0:4][3].tolist())

                            boxes.append([startX, startY, int(endX - startX), int(endY - startY)])
                            boxesXY.append([endX, endY])
                            confidences.append(float(confidence))

                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            tracker.start_track(rgb, rect)

                            classIDs3.append([classID, tracker])

                            # 인식된 객체를 추적 목록에 추가
                            trackers3.append(tracker)

            # 객체를 탐지하지 않을 때
            else:
                # 추적된 객체 수 만큼 반복
                for f, tracker in enumerate(trackers1):
                    # 추적된 객체 위치
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # bounding box 좌표 추출
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # bounding box 좌표 추가
                    rects1.append((startX, startY, endX, endY))
                    # bounding box 출력
                    cv2.rectangle(im0, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(im0, f"{names[classIDs1[f][0]]}", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor,
                                thickness=fontThickness)

                for f, tracker in enumerate(trackers2):
                    # 추적된 객체 위치
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # bounding box 좌표 추출
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # bounding box 좌표 추가
                    rects2.append((startX, startY, endX, endY))
                    # bounding box 출력
                    cv2.rectangle(im0, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(im0, f"{names[classIDs2[f][0]]}", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor,
                                thickness=fontThickness)

                for f, tracker in enumerate(trackers3):
                    # 추적된 객체 위치
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # bounding box 좌표 추출
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # bounding box 좌표 추가
                    rects3.append((startX, startY, endX, endY))
                    # bounding box 출력
                    cv2.rectangle(im0, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(im0, f"{names[classIDs3[f][0]]}", (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor,
                                thickness=fontThickness)

                # 객체 중심 추적
            objects1 = ct1.update(rects1)
            objects2 = ct2.update(rects2)
            objects3 = ct3.update(rects3)
            object_temp = [objects1, objects2, objects3]

            # 추적된 객체 수 만큼 반복
            for f, objects in enumerate(object_temp):
                for (objectID, centroid) in objects.items():
                    # table 안으로 객체 들어옴
                    if (table1[0][0] <= centroid[0] <= (table1[0][0] + table1[1][0]) / 3 * 2 and table1[0][1] <= centroid[
                        1] <= table1[1][1]):
                        table1UpInfo.append(objectID)
                        if len(table1UpInfo) > 1:
                            del table1UpInfo[0]
                        jsonTable['table1']['chair']['up'] = f+1

                    if ((table1[0][0] + table1[1][0]) / 3 * 2 <= centroid[0] <= table1[1][0] and table1[0][1] <= centroid[
                        1] <= table1[1][1]):
                        table1DownInfo.append(objectID)
                        if len(table1DownInfo) > 1:
                            del table1DownInfo[0]
                        jsonTable['table1']['chair']['down'] = f+1

                    if (table2[0][0] <= centroid[0] <= (table2[0][0] + table2[1][0]) / 2 and table2[0][1] <= centroid[1] <=
                            table2[1][1]):
                        table2UpInfo.append(objectID)
                        if len(table2UpInfo) > 1:
                            del table2UpInfo[0]
                        jsonTable['table2']['chair']['up'] = f+1

                    if ((table2[0][0] + table2[1][0]) / 2 <= centroid[0] <= table2[1][0] and table2[0][1] <= centroid[1] <=
                            table2[1][1]):
                        table2DownInfo.append(objectID)
                        if len(table2DownInfo) > 1:
                            del table2DownInfo[0]
                        jsonTable['table2']['chair']['down'] = f+1

                    if (table3[0][0] <= centroid[0] <= (table3[0][0] + table3[1][0]) / 2 and table3[0][1] <= centroid[1] <=
                            table3[1][1]):
                        table3UpInfo.append(objectID)
                        if len(table3UpInfo) > 1:
                            del table3UpInfo[0]
                        jsonTable['table3']['chair']['up'] = f+1

                    if ((table3[0][0] + table3[1][0]) / 2 <= centroid[0] <= table3[1][0] and table3[0][1] <= centroid[1] <=
                            table3[1][1]):
                        table3DownInfo.append(objectID)
                        if len(table3DownInfo) > 1:
                            del table3DownInfo[0]
                        jsonTable['table3']['chair']['down'] = f+1

                    # table 밖으로 객체 이동
                    if ((table1[0][0] > centroid[0] or centroid[0] > (table1[0][0] + table1[1][0]) / 3 * 2 or table1[0][1] >
                         centroid[1] or centroid[1] > table1[1][1]) and table1UpInfo.count(objectID) != 0):
                        table1UpInfo.remove(objectID)
                        jsonTable['table1']['chair']['up'] = 0

                    if (((table1[0][0] + table1[1][0]) / 3 * 2 > centroid[0] or centroid[0] > table1[1][0] or table1[0][1] >
                         centroid[1] or centroid[1] > table1[1][1]) and table1DownInfo.count(objectID) != 0):
                        table1DownInfo.remove(objectID)
                        jsonTable['table1']['chair']['down'] = 0

                    if ((table2[0][0] > centroid[0] or centroid[0] > (table2[0][0] + table2[1][0]) / 2 or table2[0][1] >
                         centroid[1] or centroid[1] > table2[1][1]) and table2UpInfo.count(objectID) != 0):
                        table2UpInfo.remove(objectID)
                        jsonTable['table2']['chair']['up'] = 0

                    if (((table2[0][0] + table2[1][0]) / 2 > centroid[0] or centroid[0] > table2[1][0] or table2[0][1] >
                         centroid[1] or centroid[1] > table2[1][1]) and table2DownInfo.count(objectID) != 0):
                        table2DownInfo.remove(objectID)
                        jsonTable['table2']['chair']['down'] = 0
                        print("4번 out")

                    if ((table3[0][0] > centroid[0] or centroid[0] > (table3[0][0] + table3[1][0]) / 2 or table3[0][1] >
                         centroid[1] or centroid[1] > table3[1][1]) and table3UpInfo.count(objectID) != 0):
                        table3UpInfo.remove(objectID)
                        jsonTable['table3']['chair']['up'] = 0

                    if (((table3[0][0] + table3[1][0]) / 2 > centroid[0] or centroid[0] > table3[1][0] or table3[0][1] >
                         centroid[1] or centroid[1] > table3[1][1]) and table3DownInfo.count(objectID) != 0):
                        table3DownInfo.remove(objectID)
                        jsonTable['table3']['chair']['down'] = 0

            cv2.rectangle(im0, tableBoard[0], tableBoard[1], color=(255, 255, 255), thickness=-1)

            print("jsonTable['table1']['chair']['up'] : ", jsonTable['table1']['chair']['up'])
            print("table1UpInfo : ", table1UpInfo)
            print("jsonTable['table1']['chair']['down'] : ", jsonTable['table1']['chair']['down'])
            print("table1UpInfo : ", table1DownInfo)
            print("jsonTable['table2']['chair']['up'] : ", jsonTable['table2']['chair']['up'])
            print("table1UpInfo : ", table2UpInfo)
            print("jsonTable['table2']['chair']['down'] : ", jsonTable['table2']['chair']['down'])
            print("table1UpInfo : ", table2DownInfo)
            print("jsonTable['table3']['chair']['up'] : ", jsonTable['table3']['chair']['up'])
            print("table1UpInfo : ", table3UpInfo)
            print("jsonTable['table3']['chair']['down'] : ", jsonTable['table3']['chair']['down'])
            print("table1UpInfo : ", table3DownInfo)
            print('--------------------------------------------')
            ws.send(json.dumps(jsonTable))
            cv2.putText(im0, f"{jsonTable['table1']['chair']['up']}", \
                        (int(table1[0][0] + (table1[1][0] - table1[0][0]) / 4), int((table1[0][1] + table1[1][1]) / 2)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor, thickness=fontThickness)
            cv2.putText(im0, f"{jsonTable['table1']['chair']['down']}", \
                        (int((table1[0][0] + table1[1][0]) / 2 + (table1[1][0] - table1[0][0]) / 4),
                         int((table1[0][1] + table1[1][1]) / 2)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor, thickness=fontThickness)
            cv2.putText(im0, f"{jsonTable['table2']['chair']['up']}", \
                        (int(table2[0][0] + (table2[1][0] - table2[0][0]) / 4), int((table2[0][1] + table2[1][1]) / 2)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor, thickness=fontThickness)
            cv2.putText(im0, f"{jsonTable['table2']['chair']['down']}", \
                        (int((table2[0][0] + table2[1][0]) / 2 + (table2[1][0] - table2[0][0]) / 4),
                         int((table2[0][1] + table2[1][1]) / 2)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor, thickness=fontThickness)
            cv2.putText(im0, f"{jsonTable['table3']['chair']['up']}", \
                        (int(table3[0][0] + (table3[1][0] - table3[0][0]) / 4), int((table3[0][1] + table3[1][1]) / 2)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor, thickness=fontThickness)
            cv2.putText(im0, f"{jsonTable['table3']['chair']['down']}", \
                        (int((table3[0][0] + table3[1][0]) / 2 + (table3[1][0] - table3[0][0]) / 4),
                         int((table3[0][1] + table3[1][1]) / 2)), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=fontColor, thickness=fontThickness)

            # Counting 정보를 반복
            for u, key in enumerate(jsonTable):  # key : table1, table2, table3
                things = list(jsonTable[key].keys())  # things[0] : chair, things[1] : object
                chair = jsonTable[key][things[0]]  # chair dict
                tableObjects = jsonTable[key][things[1]]  # object dict
                mask = 0
                nomask = 0
                error = 0
                tableInfo = ''

                def person(x):
                    global mask, nomask, error
                    if x == 1:
                        mask += 1
                    elif x == 2:
                        nomask += 1
                    elif x == 3:
                        error += 1

                if sum(chair.values()) == 0 and sum(tableObjects.values()) == 0:  # 사람도 없고, 물건도 없어
                    tableInfo = f"{key} : Empty"
                elif sum(chair.values()) != 0:  # 사람이 있어, 물건 상관없어
                    tableInfo = f"{key} : Full"
                    for a in chair:
                        person(chair[a])
                    if mask != 0:
                        tableInfo += f" \n - mask : {mask}"
                    if nomask != 0:
                        tableInfo += f" \n - nomask : {nomask}"
                    if error != 0:
                        tableInfo += f" \n - error : {error}"
                elif sum(chair.values()) == 0 and sum(tableObjects.values()) != 0:  # 사람이 없어, 물건 있어
                    tableInfo = f"{key} : Full"
                    for b in tableObjects:
                        if tableObjects[b] != 0:
                            tableInfo += f" \n - {b} : {tableObjects[b]}"
                for c, line in enumerate(tableInfo.split('\n')):
                    h = 15 + 20 * c
                    cv2.putText(im0, line, (15 + (u * 170), h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=fontColor,
                                thickness=fontThickness)

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

            # 총 프레임 수 증가
            totalFrames += 1

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)