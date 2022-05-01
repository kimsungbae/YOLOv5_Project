##### 실행 #####
# 비디오를 저장하지 않을 경우
# webcam : sudo python3 yolo_object_tracking_and_counting.py
# 예) sudo python3 yolo_object_tracking_and_couning.py
# video : sudo python3 yolo_object_tracking_and_couning.py --input 비디오 경로
# 예) sudo python3 yolo_object_tracking_and_couning.py --input test_video.mp4
#
# 비디오를 저장할 경우
# webcam : sudo python3 yolo_object_tracking_and_couning.py --output 저장할 비디오 경로
# 예) sudo python3 yolo_object_tracking_and_couning.py --output result_video_yolo.avi
# video : sudo python3 yolo_object_tracking_and_couning.py --input 비디오 경로 --output 저장할 비디오 경로
# 예) sudo python3 yolo_object_tracking_and_couning.py --input test_video.mp4 --output result_video_yolo.avi

# 필요한 패키지 import
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
from imutils.video import FPS
import numpy as np  # 파이썬 행렬 수식 및 수치 계산 처리 모듈
import argparse  # 명령행 파싱(인자를 입력 받고 파싱, 예외처리 등) 모듈
import imutils  # 파이썬 OpenCV가 제공하는 기능 중 복잡하고 사용성이 떨어지는 부분을 보완(이미지 또는 비디오 스트림 파일 처리 등)
import time  # 시간 처리 모듈

from datetime import datetime
# 웹소켓

import websocket
import time
from datetime import datetime
import json
from random import randint

import json
from random import randint
import dlib  # 이미지 처리 및 기계 학습, 얼굴인식 등을 할 수 있는 고성능의 라이브러리
import cv2  # opencv 모듈
import os  # 운영체제 기능 모듈

# 실행을 할 때 인자값 추가
ap = argparse.ArgumentParser()  # 인자값을 받을 인스턴스 생성
# 입력받을 인자값 등록
ap.add_argument("-i", "--input", type=str, help="input 비디오 경로")
ap.add_argument("-o", "--output", type=str, help="output 비디오 경로")  # 비디오 저장 경로
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="최소 확률")
ap.add_argument("-s", "--skip-frames", type=int, default=30, help="추적된 객체에서 다시 객체를 탐지하기까지 건너뛸 프레임 수")
# 입력받은 인자값을 args에 저장
args = vars(ap.parse_args())

# YOLO 모델이 학습된 coco 클래스 레이블
labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# YOLO 가중치 및 모델 구성에 대한 경로
# weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])  # 가중치
weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])  # 가중치
configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])  # 모델 구성

# COCO 데이터 세트(80 개 클래스)에서 훈련된 YOLO 객체 감지기 load
print("[YOLO 객체 감지기 loading...]")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# YOLO에서 필요한 output 레이어 이름
ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] # ubuntu용
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()] # local용

# input 비디오 경로가 제공되지 않은 경우 webcam
if not args.get("input", False):
    print("[webcam 시작]")
    vs = cv2.VideoCapture('rtsp://192.168.0.35:8080/h264_ulaw.sdp')
    # vs = cv2.VideoCapture(0)

# input 비디오 경로가 제공된 경우 video
else:
    print("[video 시작]")
    vs = cv2.VideoCapture(args["input"])

# 비디오 저장 변수 초기화
writer = None

# 프레임 크기 초기화(비디오에서 첫 번째 프레임을 읽는 즉시 설정)
W = None
H = None

# 중심 추적 변수
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

# 추적 객체 목록
trackers = []

# 추적 객체 ID
trackableObjects = {}

# 총 프레임 수
totalFrames = 0  # 총 프레임 수

# 총 이동 객체 수
totalRight = 0
totalLeft = 0

# fps 정보 초기화
fps = FPS().start()

# 객체 시작점 튜플
object_start_tuple = ()

# table 위치 정보
table1 = [(20, 300), (270, 560)]
table2 = [(280, 300), (590, 560)]
table3 = [(610, 300), (930, 560)]

# 테이블 바운더리 어디서부터 객체 탐지
dist = 10

# 웹소켓
ws = websocket.WebSocket()
ws.connect("ws://35.77.144.191/ws/detectData")


# table 정보
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
table1UpInfo = []
table1DownInfo = []
table2UpInfo = []
table2DownInfo = []
table3UpInfo = []
table3DownInfo = []

# 비디오 스트림 프레임 반복
while True:
    # 프레임 읽기
    ret, frame = vs.read()
    #frame 정보 = im0s
    '''
    [[[ 79  93 112]
  [ 84  98 117]
  [ 81  95 114]
  ...
  [ 94 101 107]
  [ 94 101 107]
  [ 94 101 107]]

 [[ 52  66  85]
  [ 53  67  86]
  [ 49  63  82]
  ...
  [ 94 101 107]
  [ 94 101 107]
  [ 94 101 107]]

 [[ 52  66  85]
  [ 47  61  80]
  [ 45  59  78]
  ...
  [ 94 101 107]
  [ 94 101 107]
  [ 94 101 107]]

 ...

 [[157 165 169]
  [158 166 170]
  [158 166 170]
  ...
  [148 151 156]
  [149 152 157]
  [149 152 157]]

 [[155 165 169]
  [157 167 171]
  [157 167 171]
  ...
  [146 152 156]
  [147 153 157]
  [147 153 157]]

 [[156 166 170]
  [160 170 174]
  [156 166 170]
  ...
  [143 149 153]
  [143 149 153]
  [143 149 153]]]

    '''
    # 읽은 프레임이 없는 경우 종료
    if args["input"] is not None and frame is None:
        break
    # 프레임 크기 지정
    frame = imutils.resize(frame, width=1000)
    # RGB 변환
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 프레임 크기
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # output video 설정
    if args["output"] is not None and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (W, H), True)

    # 객체 bounding box 목록
    rects = []

    # 추적된 객체에서 다시 객체를 탐지하기까지 건너뛸 프레임 수 적용
    # 객체를 탐지할 때
    if totalFrames % args["skip_frames"] == 0:
        # 객체 추적 목록 초기화
        trackers = []

        # blob 이미지 생성
        # 파라미터
        # 1) image : 사용할 이미지
        # 2) scalefactor : 이미지 크기 비율 지정
        # 3) size : Convolutional Neural Network에서 사용할 이미지 크기를 지정
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        # 객체 인식
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        boxes = []
        boxesXY = []
        confidences = []
        classIDs = []

        # layerOutputs 반복
        for output in layerOutputs: #layerOutputs는 3개츷 / 각 층에 객체 정보있음(중복 있음)
            # 각 클래스 레이블마다 인식된 객체 수 만큼 반복
            for detection in output: # detection에는 객체 하나의 정보(det)
                # 인식된 객체의 클래스 ID 및 확률 추출
                scores = detection[5:] # detection에서 앞에 5개 값빼고 추출
                classID = np.argmax(scores) # score에서 가장 큰 인덱스 변환
                confidence = scores[classID]
                # 사람인 경우
                if classID == 0 or classID == 1 or classID == 2:  # yolo-coco 디렉터리에 coco.names 파일을 참고하여 다른 object 도 인식 가능(0 인 경우 사람)
                    # 객체 확률이 최소 확률보다 큰 경우
                    if confidence > args["confidence"]:
                        # bounding box 위치 계산
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")  # (중심 좌표 X, 중심 좌표 Y, 너비(가로), 높이(세로))

                        # bounding box  좌표
                        startX = int(centerX - (width / 2))
                        startY = int(centerY - (height / 2))
                        endX = int(centerX + (width / 2))
                        endY = int(centerY + (height / 2))

                        boxes.append([startX, startY, int(width), int(height)])
                        boxesXY.append([endX, endY])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        conf_threshold = 0.4
        nms_threshold = 0.4
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        if len(idxs) > 0:
            for i in idxs.flatten():
                # 객체 추적 정보 추출
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(boxes[i][0], boxes[i][1], boxesXY[i][0], boxesXY[i][1])
                tracker.start_track(rgb, rect)

                # 인식된 객체를 추적 목록에 추가
                trackers.append(tracker)

   # 객체를 탐지하지 않을 때
    else:
        # 추적된 객체 수 만큼 반복
        for tracker in trackers:
            # 추적된 객체 위치
            tracker.update(rgb)
            pos = tracker.get_position()

            # bounding box 좌표 추출
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # bounding box 좌표 추가
            rects.append((startX, startY, endX, endY))

            # bounding box 출력
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    # 객체 중심 추적
    objects = ct.update(rects)
    # 추적된 객체 수 만큼 반복
    for (objectID, centroid) in objects.items():
        # 현재 객체 ID에 대해 추적 가능한 객체 확인
        to = trackableObjects.get(objectID, None)

        # table 안으로 객체 들어옴
        if ((table1[0][0] <= centroid[0] <= (table1[0][0] + table1[1][0]) / 3 * 2 and \
                table1[0][1] <= centroid[1] <= table1[1][1]) and \
                table1UpInfo.count(objectID) < 1):
            table1UpInfo.append(objectID)
            # to.counted = True
            jsonTable['table1']['chair']['up'] = +1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table1UpInfo : ', table1UpInfo)
        if (((table1[0][0] + table1[1][0]) / 3 * 2 <= centroid[0] <= table1[1][0] and \
                table1[0][1] <= centroid[1] <= table1[1][1]) and \
                table1DownInfo.count(objectID) < 1):
            table1DownInfo.append(objectID)
            # to.counted = True
            jsonTable['table1']['chair']['down'] += 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table1DownInfo : ', table1DownInfo)
        if ((table2[0][0] <= centroid[0] <= (table2[0][0] + table2[1][0]) / 2 and \
                table2[0][1] <= centroid[1] <= table2[1][1]) and \
                table2UpInfo.count(objectID) < 1):
            table2UpInfo.append(objectID)
            # to.counted = True
            jsonTable['table2']['chair']['up'] += 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table2UpInfo : ', table2UpInfo)
        if (((table2[0][0] + table2[1][0]) / 2 <= centroid[0] <= table2[1][0] and \
                table2[0][1] <= centroid[1] <= table2[1][1]) and \
                table2DownInfo.count(objectID) < 1):
            table2DownInfo.append(objectID)
            # to.counted = True
            jsonTable['table2']['chair']['down'] += 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table2DownInfo : ', table2DownInfo)
        if ((table3[0][0] <= centroid[0] <= (table3[0][0] + table3[1][0]) / 2 and \
                table3[0][1] <= centroid[1] <= table3[1][1]) and \
                table3UpInfo.count(objectID) < 1):
            table3UpInfo.append(objectID)
            # to.counted = True
            jsonTable['table3']['chair']['up'] += 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table3UpInfo : ', table3UpInfo)
            print('table3UpInfo.count(objectID) : ', table3UpInfo.count(objectID))
        if (((table3[0][0] + table3[1][0]) / 2 <= centroid[0] <= table3[1][0] and \
                table3[0][1] <= centroid[1] <= table3[1][1]) and \
                table3DownInfo.count(objectID) < 1):
            table3DownInfo.append(objectID)
            # to.counted = True
            jsonTable['table3']['chair']['down'] += 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table3DownInfo : ', table3DownInfo)

        # table 밖으로 객체 이동
        if ((table1[0][0] > centroid[0] or centroid[0] > (table1[0][0] + table1[1][0]) / 3 * 2 or \
             table1[0][1] > centroid[1] or centroid[1] > table1[1][1]) and \
                table1UpInfo.count(objectID) != 0):
            table1UpInfo.remove(objectID)
            # to.counted = True
            jsonTable['table1']['chair']['up'] -= 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table1UpInfo : ', table1UpInfo)
        if (((table1[0][0] + table1[1][0]) / 3 * 2 > centroid[0] or centroid[0] > table1[1][0] or \
             table1[0][1] > centroid[1] or centroid[1] > table1[1][1]) and \
                table1DownInfo.count(objectID) != 0):
            table1DownInfo.remove(objectID)
            # to.counted = True
            jsonTable['table1']['chair']['down'] -= 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table1DownInfo : ', table1DownInfo)
        if ((table2[0][0] > centroid[0] or centroid[0] > (table2[0][0] + table2[1][0]) / 2 or \
             table2[0][1] > centroid[1] or centroid[1] > table2[1][1]) and \
                table2UpInfo.count(objectID) != 0):
            table2UpInfo.remove(objectID)
            # to.counted = True
            jsonTable['table2']['chair']['up'] -= 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table2UpInfo : ', table2UpInfo)
        if (((table2[0][0] + table2[1][0]) / 2 > centroid[0] or centroid[0] > table2[1][0] or \
             table2[0][1] > centroid[1] or centroid[1] > table2[1][1]) and \
                table2DownInfo.count(objectID) != 0):
            table2DownInfo.remove(objectID)
            # to.counted = True
            jsonTable['table2']['chair']['down'] -= 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table2DownInfo : ', table2DownInfo)
        if ((table3[0][0] > centroid[0] or centroid[0] > (table3[0][0] + table3[1][0]) / 2 or \
             table3[0][1] > centroid[1] or centroid[1] > table3[1][1]) and \
                table3UpInfo.count(objectID) != 0):
            table3UpInfo.remove(objectID)
            # to.counted = True
            jsonTable['table3']['chair']['up'] -= 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table3UpInfo : ', table3UpInfo)
        if (((table3[0][0] + table3[1][0]) / 2 > centroid[0] or centroid[0] > table3[1][0] or \
             table3[0][1] > centroid[1] or centroid[1] > table3[1][1]) and \
                table3DownInfo.count(objectID) != 0):
            table3DownInfo.remove(objectID)
            # to.counted = True
            jsonTable['table3']['chair']['down'] -= 1
            print('objectID : ', objectID)
            print('centroid[0] ; ', centroid[0])
            print('centroid[1] ; ', centroid[1])
            print('table3DownInfo : ', table3DownInfo)


        # 추적 가능한 객체가 없는 경우
        # if to is None:
        #     # 하나의 객체를 생성
        #     to = TrackableObject(objectID, centroid)
        #     print('추적 가능한 객체 없음')
        #     print('to, :', to)
        #
        # # 추적 가능한 객체가 있는 경우
        # else:
        #     print('추적 가능한 객체가 있어???????')
        #     print(table3[0][0] > centroid[0])
        #     print(centroid[0] > (table3[0][0] + table3[1][0]) / 2)
        #     print(table3[0][1] > centroid[1])
        #     print(centroid[1] > table3[1][1])
        #     print(table3UpInfo.count(objectID) != 0)

            # # 이전의 중심 좌표에 대한 가로 좌표 값을 추출
            # y = [c[0] for c in to.centroids]
            # # 이전의 중심 좌표에 대한 세로 좌표 값을 추출
            # z = [c[1] for c in to.centroids]
            #
            # # 현재 중심 좌표와 이전 중심 좌표의 평균의 차이를 이용하여 가로 방향을 계산
            # direction_horizontal = centroid[0] - np.mean(y) # x좌표
            # # 현재 중심 좌표와 이전 중심 좌표의 평균의 차이를 이용하여 세로 방향을 계산
            # direction_vertical = centroid[1] - np.mean(z) # y좌표
            #
            # # 중심 좌표 추가
            # to.centroids.append(centroid)

            # 객체가 counting 되지 않았을 때
            # if not to.counted:
                # # 어느방향으로 이동하냐
                # #  - table 바운더리안에 들어올때 or 나갈때
                # #  - table1 ,2, 3
                # #  -
                # # 1) x좌표 : 왼쪽 -> 오른쪽 (y좌표 : 위쪽과 아래쪽 사이)
                # # 2) x좌표 : 오른쪽 -> 왼쪽 (y좌표 : 위쪽과 아래쪽 사이)
                # # 3) y좌표 : 위쪽 -> 아래쪽 (x좌표 : 왼쪽과 오른쪽 사이)
                # # 4) y좌표 : 아래쪽 -> 위쪽 (x좌표 : 왼쪽과 오른쪽 사이)
                # #
                # # 1) x좌표 : 왼쪽 -> 오른쪽 (y좌표 : 위쪽과 아래쪽 사이)
                # #  - 바운더리 들어올때, 가운데, 나갈때
                # #  - table1, 2, 3
                # #  - dist는 바운더리 근처 어디서부터 확인할 지, 대각선에서 오는것도 확인하기 위해서
                # if ((table1[0][1] - dist <= centroid[1] <= table1[1][1] + dist) and \
                #     ((centroid[0] <= table1[0][0] - dist) or \
                #      (centroid[0] <= (table1[0][0] + table1[1][0]) / 2 - dist) or \
                #      (centroid[0] <= table1[1][0] - dist))) or \
                #         ((table2[0][1] - dist <= centroid[1] <= table2[1][1] + dist) and \
                #          ((centroid[0] <= table2[0][0] - dist) or \
                #           (centroid[0] <= (table2[0][0] + table2[1][0]) / 2 - dist) or \
                #           (centroid[0] <= table2[1][0] - dist))) or \
                #         ((table3[0][1] - dist <= centroid[1] <= table3[1][1] + dist) and \
                #          ((centroid[0] <= table3[0][0] - dist) or \
                #           (centroid[0] <= (table3[0][0] + table3[1][0]) / 2 - dist) or \
                #           (centroid[0] <= table3[1][0] - dist))):
                #     try:
                #         if len(object_start_tuple) < objectID:
                #             object_start_tuple = object_start_tuple + (0,)
                #         elif len(object_start_tuple) == objectID:
                #             object_start_tuple = object_start_tuple + (-1,)
                #     except:
                #         pass
                #
                # # 2) x좌표 : 오른쪽 -> 왼쪽 (y좌표 : 위쪽과 아래쪽 사이)
                # #  - 바운더리 들어올때, 가운데, 나갈때
                # #  - table1, 2, 3
                # #  - dist는 바운더리 근처 어디서부터 확인할 지, 대각선에서 오는것도 확인하기 위해서
                # elif ((centroid[1] >= table1[0][1] - dist) and (centroid[1] <= table1[1][1] + dist) and \
                #       ((centroid[0] >= table1[0][0] + dist) or \
                #        (centroid[0] >= (table1[0][0] + table1[1][0]) / 2 + dist) or \
                #        (centroid[0] >= table1[1][0] + dist))) or \
                #      ((centroid[1] >= table2[0][1] - dist) and (centroid[1] <= table2[1][1] + dist) and \
                #       ((centroid[0] >= table2[0][0] + dist) or \
                #        (centroid[0] >= (table2[0][0] + table2[1][0]) / 2 + dist) or \
                #        (centroid[0] >= table2[1][0] + dist))) or \
                #      ((centroid[1] >= table3[0][1] - dist) and (centroid[1] <= table3[1][1] + dist) and \
                #       ((centroid[0] >= table3[0][0] + dist) or \
                #        (centroid[0] >= (table3[0][0] + table3[1][0]) / 2 + dist) or \
                #        (centroid[0] >= table3[1][0] + dist))):
                #     try:
                #         if len(object_start_tuple) < objectID:
                #             object_start_tuple = object_start_tuple + (0,)
                #         elif len(object_start_tuple) == objectID:
                #             object_start_tuple = object_start_tuple + (1,)
                #     except:
                #         pass
                #
                # # 3) y좌표 : 위쪽 -> 아래쪽 (x좌표 : 왼쪽과 오른쪽 사이)
                # #  - 바운더리 들어올때, 나갈때
                # #  - table1, 2, 3
                # #  - dist는 바운더리 근처 어디서부터 확인할 지, 대각선에서 오는것도 확인하기 위해서
                # elif ((centroid[0] >= table1[0][0] - dist) and (centroid[0] <= table1[1][0] + dist) and \
                #       ((centroid[1] <= table1[0][1] - dist) or \
                #        (centroid[1] <= table1[1][1] - dist))) or \
                #      ((centroid[0] >= table2[0][0] - dist) and (centroid[0] <= table2[1][0] + dist) and \
                #       ((centroid[1] <= table2[0][1] - dist) or \
                #        (centroid[1] <= table2[1][1] - dist))) or \
                #      ((centroid[0] >= table3[0][0] - dist) and (centroid[0] <= table3[1][0] + dist) and \
                #       ((centroid[1] <= table3[0][1] - dist) or \
                #        (centroid[1] <= table3[1][1] - dist))):
                #     try:
                #         if len(object_start_tuple) < objectID:
                #             object_start_tuple = object_start_tuple + (0,)
                #         elif len(object_start_tuple) == objectID:
                #             object_start_tuple = object_start_tuple + (-2,)
                #     except:
                #         pass
                #
                # # 4) y좌표 : 아래쪽 -> 위쪽 (x좌표 : 왼쪽과 오른쪽 사이)
                # #  - 바운더리 들어올때, 나갈때
                # #  - table1, 2, 3
                # #  - dist는 바운더리 근처 어디서부터 확인할 지, 대각선에서 오는것도 확인하기 위해서
                # elif ((centroid[0] >= table1[0][0] - dist) and (centroid[0] <= table1[1][0] + dist) and \
                #       ((centroid[1] >= table1[0][1] + dist) or \
                #        (centroid[1] >= table1[1][1] + dist))) or \
                #      ((centroid[0] >= table2[0][0] - dist) and (centroid[0] <= table2[1][0] + dist) and \
                #       ((centroid[1] >= table2[0][1] + dist) or \
                #        (centroid[1] >= table2[1][1] + dist))) or \
                #      ((centroid[0] >= table3[0][0] - dist) and (centroid[0] <= table3[1][0] + dist) and \
                #       ((centroid[1] >= table3[0][1] + dist) or \
                #        (centroid[1] >= table3[1][1] + dist))):
                #     try:
                #         if len(object_start_tuple) < objectID:
                #             object_start_tuple = object_start_tuple + (0,)
                #         elif len(object_start_tuple) == objectID:
                #             object_start_tuple = object_start_tuple + (2,)
                #     except:
                #         pass
                #
                # try:
                    # # talbe1 / up / in
                    # # 범위 안에 in(왼쪽보다 크고, 오른쪽보다 작고, 위쪽보다 크고, 아래쪽보다 작다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (table1[0][0] <= centroid[0] <= (table1[0][0] + table1[1][0]) / 2 and \
                    #     table1[0][1] <= centroid[1] <= table1[1][1]) and \
                    #     ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #      (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #      (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #      (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table1']['chair']['up'] = 1
                    #
                    # # talbe1 / up / out
                    # # 범위 밖으로 out(왼쪽보다 작고, 오른쪽보다 크고, 위쪽보다 작고, 아래쪽보다 크다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (jsonTable['table1']['chair']['up'] > 0 and \
                    #     table1[0][0] > centroid[0] or centroid[0] > (table1[0][0] + table1[1][0]) / 2 or \
                    #     table1[0][1] > centroid[1] or centroid[1] > table1[1][1]) and \
                    #     ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #      (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #      (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #      (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table1']['chair']['up'] = 0
                    #
                    # # talbe1 / down / in
                    # # 범위 안에 in(왼쪽보다 크고, 오른쪽보다 작고, 위쪽보다 크고, 아래쪽보다 작다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if ((table1[0][0] + table1[1][0]) / 2 <= centroid[0] <= table1[1][0] and \
                    #     table1[0][1] <= centroid[1] <= table1[1][1]) and \
                    #     ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #      (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #      (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #      (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table1']['chair']['down'] = 1
                    #
                    # # talbe1 / down / out
                    # # 범위 밖으로 out(왼쪽보다 작고, 오른쪽보다 크고, 위쪽보다 작고, 아래쪽보다 크다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (jsonTable['table1']['chair']['down'] > 0 and \
                    #     (table1[0][0] + table1[1][0]) / 2 > centroid[0] or centroid[0] > table1[1][0] or \
                    #     table1[0][1] > centroid[1] or centroid[1] > table1[1][1]) and \
                    #     ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #      (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #      (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #      (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table1']['chair']['down'] = 0
                    #
                    # # table2 / up / in
                    # # 범위 안에 in(왼쪽보다 크고, 오른쪽보다 작고, 위쪽보다 크고, 아래쪽보다 작다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (table2[0][0] <= centroid[0] <= (table2[0][0] + table2[1][0]) / 2 and \
                    #     table2[0][1] <= centroid[1] <= table2[1][1]) and \
                    #     ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #      (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #      (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #      (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table2']['chair']['up'] = 1
                    #
                    # # table2 / up / out
                    # # 범위 밖으로 out(왼쪽보다 작고, 오른쪽보다 크고, 위쪽보다 작고, 아래쪽보다 크다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (jsonTable['table2']['chair']['up'] > 0 and \
                    #     table2[0][0] > centroid[0] or centroid[0] > (table2[0][0] + table2[1][0]) / 2 or \
                    #     table2[0][1] > centroid[1] or centroid[1] > table2[1][1]) and \
                    #         ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #          (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #          (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #          (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table2']['chair']['up'] = 0
                    #
                    # # table2 / down / in
                    # # 범위 안에 in(왼쪽보다 크고, 오른쪽보다 작고, 위쪽보다 크고, 아래쪽보다 작다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if ((table2[0][0] + table2[1][0]) / 2 <= centroid[0] <= table2[1][0] and \
                    #     table2[0][1] <= centroid[1] <= table2[1][1]) and \
                    #         ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #          (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #          (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #          (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table2']['chair']['down'] = 1
                    #
                    # # table2 / down / out
                    # # 범위 밖으로 out(왼쪽보다 작고, 오른쪽보다 크고, 위쪽보다 작고, 아래쪽보다 크다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (jsonTable['table2']['chair']['down'] > 0 and \
                    #     (table2[0][0] + table2[1][0]) / 2 > centroid[0] or centroid[0] > table2[1][0] or \
                    #     table2[0][1] > centroid[1] or centroid[1] > table2[1][1]) and \
                    #         ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #          (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #          (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #          (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table2']['chair']['down'] = 0
                    #
                    # # table3 / up / in
                    # # 범위 안에 in(왼쪽보다 크고, 오른쪽보다 작고, 위쪽보다 크고, 아래쪽보다 작다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (table3[0][0] <= centroid[0] <= (table3[0][0] + table3[1][0]) / 2 and \
                    #     table3[0][1] <= centroid[1] <= table3[1][1]) and \
                    #         ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #          (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #          (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #          (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table3']['chair']['up'] = 1
                    #     print('table3 chair up +1 objectID : ', objectID)
                    #     print('table3[0][0] : ', table3[0][0])
                    #     print('centroid[0] : ', centroid[0])
                    #     print('(table3[0][0] + table3[1][0]) / 2 : ', (table3[0][0] + table3[1][0]) / 2)
                    #     print('table3[0][1] : ', table3[0][1])
                    #     print('centroid[1] : ', centroid[1])
                    #     print('table3[1][1] : ', table3[1][1])
                    #
                    # # table3 / up / out
                    # # 범위 밖으로 out(왼쪽보다 작고, 오른쪽보다 크고, 위쪽보다 작고, 아래쪽보다 크다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (jsonTable['table3']['chair']['up'] > 0 and \
                    #     table3[0][0] > centroid[0] or centroid[0] > (table3[0][0] + table3[1][0]) / 2 or \
                    #     table3[0][1] > centroid[1] or centroid[1] > table3[1][1]) and \
                    #         ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #          (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #          (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #          (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table3']['chair']['up'] = 0
                    #     print('table3 chair up -1 objectID : ', objectID)
                    #     print('table3[0][0] : ', table3[0][0])
                    #     print('centroid[0] : ', centroid[0])
                    #     print('table3[0][0] > centroid[0] : ', table3[0][0] > centroid[0])
                    #     print('centroid[0] : ', centroid[0])
                    #     print('(table3[0][0] + table3[1][0]) / 2 : ', (table3[0][0] + table3[1][0]) / 2)
                    #     print('centroid[0] > (table3[0][0] + table3[1][0]) / 2 : ', centroid[0] > (table3[0][0] + table3[1][0]) / 2)
                    #     print('table3[0][1] : ', table3[0][1])
                    #     print('centroid[1] : ', centroid[1])
                    #     print('table3[0][1] > centroid[1] : ', table3[0][1] > centroid[1])
                    #     print('centroid[1] : ', centroid[1])
                    #     print('table3[1][1] : ', table3[1][1])
                    #     print('centroid[1] > table3[1][1]) : ', centroid[1] > table3[1][1])
                    #
                    #
                    # # table3 / down / in
                    # # 범위 안에 in(왼쪽보다 크고, 오른쪽보다 작고, 위쪽보다 크고, 아래쪽보다 작다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if ((table3[0][0] + table3[1][0]) / 2 <= centroid[0] <= table3[1][0] and \
                    #     table3[0][1] <= centroid[1] <= table3[1][1]) and \
                    #         ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #          (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #          (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #          (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table3']['chair']['down'] = 1
                    #     print('table3 chair down +1 objectID : ', objectID)
                    #     print('(table3[0][0] + table3[1][0]) / 2 : ', (table3[0][0] + table3[1][0]) / 2)
                    #     print('centroid[0] <= table3[1][0] : ', centroid[0] <= table3[1][0])
                    #     print('(table3[0][0] + table3[1][0]) / 2 <= centroid[0] <= table3[1][0] : ', (table3[0][0] + table3[1][0]) / 2 <= centroid[0] <= table3[1][0])
                    #     print('centroid[0] : ', centroid[0])
                    #     print('(table3[0][0] + table3[1][0]) / 2 : ', (table3[0][0] + table3[1][0]) / 2)
                    #     print('centroid[0] > (table3[0][0] + table3[1][0]) / 2 : ',
                    #           centroid[0] > (table3[0][0] + table3[1][0]) / 2)
                    #     print('table3[0][1] : ', table3[0][1])
                    #     print('centroid[1] : ', centroid[1])
                    #     print('table3[0][1] > centroid[1] : ', table3[0][1] > centroid[1])
                    #     print('centroid[1] : ', centroid[1])
                    #     print('table3[1][1] : ', table3[1][1])
                    #     print('centroid[1] > table3[1][1]) : ', centroid[1] > table3[1][1])
                    #
                    # # table3 / down / out
                    # # 범위 밖으로 out(왼쪽보다 작고, 오른쪽보다 크고, 위쪽보다 작고, 아래쪽보다 크다)
                    # #  1) 왼쪽 -> 오른쪽 / 오른쪽으로 수평방향 이동
                    # #  2) 오른쪽 -> 왼쪽 / 왼쪽으로 수평방향 이동
                    # #  3) 위쪽 -> 아래족 / 아래쪽으로 수직방향 이동
                    # #  4) 아래쪽 -> 위쪽 / 위쪽으로 수직방향 이동
                    # if (jsonTable['table3']['chair']['down'] > 0 and \
                    #     (table3[0][0] + table3[1][0]) / 2 > centroid[0] or centroid[0] > table3[1][0] or \
                    #     table3[0][1] > centroid[1] or centroid[1] > table3[1][1]) and \
                    #         ((object_start_tuple[objectID] == -1 and direction_horizontal > 0) or \
                    #          (object_start_tuple[objectID] == 1 and direction_horizontal < 0) or \
                    #          (object_start_tuple[objectID] == -2 and direction_vertical > 0) or \
                    #          (object_start_tuple[objectID] == 2 and direction_vertical < 0)):
                    #     to.counted = True
                    #     jsonTable['table3']['chair']['down'] = 0
                    #     print('table3 chair down -1 objectID : ', objectID)
                    #     print('table3[0][0] : ', table3[0][0])
                    #     print('centroid[0] : ', centroid[0])
                    #     print('table3[0][0] > centroid[0] : ', table3[0][0] > centroid[0])
                    #     print('centroid[0] : ', centroid[0])
                    #     print('(table3[0][0] + table3[1][0]) / 2 : ', (table3[0][0] + table3[1][0]) / 2)
                    #     print('centroid[0] > (table3[0][0] + table3[1][0]) / 2 : ',
                    #           centroid[0] > (table3[0][0] + table3[1][0]) / 2)
                    #     print('table3[0][1] : ', table3[0][1])
                    #     print('centroid[1] : ', centroid[1])
                    #     print('table3[0][1] > centroid[1] : ', table3[0][1] > centroid[1])
                    #     print('centroid[1] : ', centroid[1])
                    #     print('table3[1][1] : ', table3[1][1])
                    #     print('centroid[1] > table3[1][1]) : ', centroid[1] > table3[1][1])
                #
                #
                #
                # except:
                #     print("error")
                #     print('to.counted', to.counted)
                #     pass

            # # table 안으로 객체 들어옴
            # if (table1[0][0] <= centroid[0] <= (table1[0][0] + table1[1][0]) / 2 and \
            #         table1[0][1] <= centroid[1] <= table1[1][1]):
            #     table1UpInfo.append(objectID)
            #     to.counted = True
            #     jsonTable['table1']['chair']['up'] = 1
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table1UpInfo : ', table1UpInfo)
            # if ((table1[0][0] + table1[1][0]) / 2 <= centroid[0] <= table1[1][0] and \
            #         table1[0][1] <= centroid[1] <= table1[1][1]):
            #     table1DownInfo.append(objectID)
            #     to.counted = True
            #     jsonTable['table1']['chair']['down'] = 1
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table1DownInfo : ', table1DownInfo)
            # if (table2[0][0] <= centroid[0] <= (table2[0][0] + table2[1][0]) / 2 and \
            #         table2[0][1] <= centroid[1] <= table2[1][1]):
            #     table2UpInfo.append(objectID)
            #     to.counted = True
            #     jsonTable['table2']['chair']['up'] = 1
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table2UpInfo : ', table2UpInfo)
            # if ((table2[0][0] + table2[1][0]) / 2 <= centroid[0] <= table2[1][0] and \
            #         table2[0][1] <= centroid[1] <= table2[1][1]):
            #     table2DownInfo.append(objectID)
            #     to.counted = True
            #     jsonTable['table2']['chair']['down'] = 1
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table2DownInfo : ', table2DownInfo)
            # if (table3[0][0] <= centroid[0] <= (table3[0][0] + table3[1][0]) / 2 and \
            #         table3[0][1] <= centroid[1] <= table3[1][1]):
            #     table3UpInfo.append(objectID)
            #     to.counted = True
            #     jsonTable['table3']['chair']['up'] = 1
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table3UpInfo : ', table3UpInfo)
            #     print('table3UpInfo.count(objectID) : ', table3UpInfo.count(objectID))
            # if ((table3[0][0] + table3[1][0]) / 2 <= centroid[0] <= table3[1][0] and \
            #         table3[0][1] <= centroid[1] <= table3[1][1]):
            #     table3DownInfo.append(objectID)
            #     to.counted = True
            #     jsonTable['table3']['chair']['down'] = 1
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table3DownInfo : ', table3DownInfo)
            #
            # # table 밖으로 객체 이동
            # if ((table1[0][0] > centroid[0] or centroid[0] > (table1[0][0] + table1[1][0]) / 2 or \
            #      table1[0][1] > centroid[1] or centroid[1] > table1[1][1]) and \
            #         table1UpInfo.count(objectID) != 0):
            #     table1UpInfo.remove(objectID)
            #     to.counted = True
            #     jsonTable['table1']['chair']['up'] = 0
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table1UpInfo : ', table1UpInfo)
            # if (((table1[0][0] + table1[1][0]) / 2 > centroid[0] or centroid[0] > table1[1][0] or \
            #      table1[0][1] > centroid[1] or centroid[1] > table1[1][1]) and \
            #         table1DownInfo.count(objectID) != 0):
            #     table1DownInfo.remove(objectID)
            #     to.counted = True
            #     jsonTable['table1']['chair']['down'] = 0
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table1DownInfo : ', table1DownInfo)
            # if ((table2[0][0] > centroid[0] or centroid[0] > (table2[0][0] + table2[1][0]) / 2 or \
            #      table2[0][1] > centroid[1] or centroid[1] > table2[1][1]) and \
            #         table2UpInfo.count(objectID) != 0):
            #     table2UpInfo.remove(objectID)
            #     to.counted = True
            #     jsonTable['table2']['chair']['up'] = 0
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table2UpInfo : ', table2UpInfo)
            # if (((table2[0][0] + table2[1][0]) / 2 > centroid[0] or centroid[0] > table2[1][0] or \
            #      table2[0][1] > centroid[1] or centroid[1] > table2[1][1]) and \
            #         table2DownInfo.count(objectID) != 0):
            #     table2DownInfo.remove(objectID)
            #     to.counted = True
            #     jsonTable['table2']['chair']['down'] = 0
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table2DownInfo : ', table2DownInfo)
            # if ((table3[0][0] > centroid[0] or centroid[0] > (table3[0][0] + table3[1][0]) / 2 or \
            #      table3[0][1] > centroid[1] or centroid[1] > table3[1][1]) and \
            #         table3UpInfo.count(objectID) != 0):
            #     table3UpInfo.remove(objectID)
            #     to.counted = True
            #     jsonTable['table3']['chair']['up'] = 0
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table3UpInfo : ', table3UpInfo)
            # if (((table3[0][0] + table3[1][0]) / 2 > centroid[0] or centroid[0] > table3[1][0] or \
            #      table3[0][1] > centroid[1] or centroid[1] > table3[1][1]) and \
            #         table3DownInfo.count(objectID) != 0):
            #     table3DownInfo.remove(objectID)
            #     to.counted = True
            #     jsonTable['table3']['chair']['down'] = 0
            #     print('objectID : ', objectID)
            #     print('centroid[0] ; ', centroid[0])
            #     print('centroid[1] ; ', centroid[1])
            #     print('table3DownInfo : ', table3DownInfo)

        # 추적 가능한 객체 저장
        trackableObjects[objectID] = to

        # 객체 ID
        # text = "ID {}".format(objectID)

        # 객체 ID 출력
        # cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 객체 중심 좌표 출력
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -2)

    # table 및 tableBoard 표시
    tableColor = (0, 0, 255)
    fontColor = (0, 255, 0)
    tableThickness = 2
    fontThickness = 2
    tableBoardColor = (255, 255, 255)
    tableBoardThickness = -1

    cv2.rectangle(frame, table1[0], table1[1], color=tableColor, thickness=tableThickness)
    cv2.rectangle(frame, table2[0], table2[1], color=tableColor, thickness=tableThickness)
    cv2.rectangle(frame, table3[0], table3[1], color=tableColor, thickness=tableThickness)
    cv2.rectangle(frame, tableBoard[0], tableBoard[1], color=tableBoardColor, thickness=tableBoardThickness)

    # print("jsonTable['table1']['chair']['up'] : ", jsonTable['table1']['chair']['up'])
    # print("jsonTable['table1']['chair']['down'] : ", jsonTable['table1']['chair']['down'])
    # print("jsonTable['table2']['chair']['up'] : ", jsonTable['table2']['chair']['up'])
    # print("jsonTable['table2']['chair']['down'] : ", jsonTable['table2']['chair']['down'])
    # print("jsonTable['table3']['chair']['up'] : ", jsonTable['table3']['chair']['up'])
    # print("jsonTable['table3']['chair']['down'] : ", jsonTable['table3']['chair']['down'])
    # ws.send(json.dumps(jsonTable))
    cv2.putText(frame, f"{jsonTable['table1']['chair']['up']}", \
                (int(table1[0][0] + (table1[1][0] - table1[0][0]) / 4), int((table1[0][1] + table1[1][1]) / 2)), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{jsonTable['table1']['chair']['down']}", \
                (int((table1[0][0] + table1[1][0]) / 2 + (table1[1][0] - table1[0][0]) / 4),
                 int((table1[0][1] + table1[1][1]) / 2)), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{jsonTable['table2']['chair']['up']}", \
                (int(table2[0][0] + (table2[1][0] - table2[0][0]) / 4), int((table2[0][1] + table2[1][1]) / 2)), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{jsonTable['table2']['chair']['down']}", \
                (int((table2[0][0] + table2[1][0]) / 2 + (table2[1][0] - table2[0][0]) / 4),
                 int((table2[0][1] + table2[1][1]) / 2)), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{jsonTable['table3']['chair']['up']}", \
                (int(table3[0][0] + (table3[1][0] - table3[0][0]) / 4), int((table3[0][1] + table3[1][1]) / 2)), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f"{jsonTable['table3']['chair']['down']}", \
                (int((table3[0][0] + table3[1][0]) / 2 + (table3[1][0] - table3[0][0]) / 4),
                 int((table3[0][1] + table3[1][1]) / 2)), \
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Counting 정보를 반복
    for i, key in enumerate(jsonTable):  # key : table1, table2, table3
        things = list(jsonTable[key].keys())  # things[0] : chair, things[1] : object
        chair = jsonTable[key][things[0]]  # chair dict
        objects = jsonTable[key][things[1]]  # object dict
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


        if sum(chair.values()) == 0 and sum(objects.values()) == 0:  # 사람도 없고, 물건도 없어
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
        elif sum(chair.values()) == 0 and sum(objects.values()) != 0:  # 사람이 없어, 물건 있어
            tableInfo = f"{key} : Full"
            for b in objects:
                if objects[b] != 0:
                    tableInfo += f" \n - {b} : {objects[b]}"
        for c, line in enumerate(tableInfo.split('\n')):
            h = 15 + 20*c
            cv2.putText(frame, line, (15 + (i*170), h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=fontColor, thickness=fontThickness)


        # Counting 정보 출력
        # cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color = fontColor, thickness = fontThickness)
       
        # cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=fontColor, thickness=fontThickness)
        # print(text)
    # # 비디오 저장
    # if writer is not None:
    #     writer.write(frame)

    # 프레임 출력
    cv2.imshow("People Tracking and Counting", frame)
    key = cv2.waitKey(10) & 0xFF

    # 'q' 키를 입력하면 종료
    if key == ord("q"):
        break

    # 총 프레임 수 증가
    totalFrames += 1

    # fps 정보 업데이트
    fps.update()

# fps 정지 및 정보 출력
fps.stop()
print("[재생 시간 : {:.2f}초]".format(fps.elapsed()))
print("[FPS : {:.2f}]".format(fps.fps()))

# 비디오 저장 종료
if writer is not None:
    writer.release()

# 종료
vs.release()
cv2.destroyAllWindows()