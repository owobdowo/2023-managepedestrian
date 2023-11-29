
from persondetection import DetectorAPI
from imutils.video import VideoStream #for usbcam or RBPi camera module
from flask import Response
from flask import Flask
from flask import render_template
import threading #for multipile clients,webbrowsers,tabs at the same time
import time
import cv2
from flask import Flask, render_template, Response
import schedule
import time



app = Flask(__name__)

odapi = DetectorAPI()


#영상 여러 개 삽입하기 위한 dic

dic_path={"./street_video0.MP4":0,"./street_video1.MP4":1,"./street_video2.MP4":2,"./street_video3.MP4":3} #키값 : 주소 value :num
dic_num={}

#일정 간격
record_interval=10 #상수, 기록 단위(초) 60>10으로 수정
#동기화될 숫자
#각 영상 사람 수 배열,실시간 사람 수, 일정 간격 사람 수 합, 일정 간격 사람수 평균
person_num=[5 for i in range(len(dic_path))]
person_sum=[5 for i in range(len(dic_path))]#60초마다 초기화
person_avg=[person_sum[i]/record_interval for i in range(len(dic_path))] #60초마다 초기화]
county=[[5 for i in range(24)] for j in range(len(dic_path))] #사람 수 통계, county[]

current_frame = None
lock = threading.Lock()

def capture_frame(path):
    global current_frame
    camera = cv2.VideoCapture(path)  # 카메라 장치 번호 (일반적으로 0)

    success, frame = camera.read()
    if success:
        with lock:
            current_frame = frame
# main
def generate_frames(path):
    while True:
        camera = cv2.VideoCapture(path)  # 카메라 장치 번호 (일반적으로 0)

        # 카메라에서 프레임 읽기
        success, frame = camera.read() #frame캡처
        frame_rate = camera.get(cv2.CAP_PROP_FPS)#프레임 속도 알기
        interval= int(frame_rate)#캡처단위=1초 더 하고 싶으면 *n하셈
        frame_counter=0
        record_num=0
        
        threshold = 0.7
        if not success:
            break
        else:
            with lock :   
                county = []
                #시간 축
                      
                frame = cv2.resize(frame, (800, 600)) #frame 리사이징 해서 넣음
                boxes, scores, classes, num = odapi.processFrame(frame) #검출기 가동#############
                person = 0

                for i in range(len(boxes)): #박스 그리기

                    if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        person += 1
                        cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                        cv2.putText(frame, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
               
               #총 사람 수 프린트
                text = "{} : {}".format("person", person)
                (H, W) = frame.shape[:2]
                cv2.putText(frame,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

#
                #사람 수 기록
                person_num=person
                county.append(person) #사람 수 셌던 기록



                # 프레임 스트리밍을 위해 웹으로 전송
                ret, buffer = cv2.imencode('.jpg', frame)#('.jpg',frame,[압축품질])로 압축해서 출력속도 늘리기 아니면 이미지크기 갑 낮을 수록 압축높음
                frame = buffer.tobytes() #문자열로 변환후 웹으로 전송
                
                # 프레임 반환
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/') #기본 서버 페이지
def index():
    # 웹 페이지 템플릿 표시
    return render_template('index.html')#해당 템플릿 호출


@app.route('/video_feed0')
def video_feed0():
    # 영상 스트리밍을 위한 Response 객체 반환
    return Response(generate_frames("./street_video0.MP4"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/human_data')
def get_person_num():
    # 정수 데이터 생성
    data = person_num

    # 데이터를 문자열로 변환하여 전송
    return Response(str(data), mimetype='text/plain')


if __name__ == '__main__':
    # 스레드를 사용하여 프레임을 일정 간격으로 캡처
    # schedule.every(0.1).seconds.do(capture_frame)
    app.run('0.0.0.0',port=5000,debug=True)

    while True:
        schedule.run_pending()
        time.sleep(0.1)

        #실시간으로 되긴하는데 작업복잡도때문에 텀이 좀 길다
        #컴퓨터 문제 일수도...
    