
from persondetection import DetectorAPI
from imutils.video import VideoStream #for usbcam or RBPi camera module
import threading #for multipile clients,webbrowsers,tabs at the same time
from flask import Response
from flask import Flask
from flask import render_template
from flask import jsonify
import cv2
import schedule
import time
import jsonify

##path설정시 dict 변경, 
app = Flask(__name__)

odapi = DetectorAPI()


#영상 여러 개 삽입하기 위한 dic

dic_path={"./street_video0.MP4":0,"./street_video1.MP4":1,"./street_video2.MP4":2,"./street_video3.MP4":3} #키값 : 주소 value :num
dic_num={}

#일정 간격
record_interval=10 #상수, 기록 단위(초) 60>10으로 수정
#동기화될 숫자
#각 영상 사람 수 배열,실시간 사람 수, 일정 간격 사람 수 합, 일정 간격 사람수 평균
county=[[2 for i in range(24)] for j in range(len(dic_path))] 
person_num=[2 for i in range(len(dic_path))]#경로 길이 만큼의 사람 수...
person_sum=[2 for i in range(len(dic_path))]#60초마다 초기화
person_avg=[2 for i in range(len(dic_path))] #60초마다 초기화]

current_frame = None
lock = threading.Lock()

def capture_frame(path):
    global current_frame
    camera = cv2.VideoCapture(path)  # 카메라 장치 번호 (일반적으로 0)

    success, frame = camera.read()
    if success:
        with lock:
            current_frame = frame
def frame_processing(path,frame,MIrecording,record_num): #path: dict용, 이차원 배열에 등록할 때 필요, frame:처리대상, MIrecording recording해야하는지,record_num: 몇신지
    global person_num
    global person_sum
    global person_avg
    global county
    global record_interval
    global dic_path

    
    person = 0
    threshold = 0.7

    frame = cv2.resize(frame, (800, 600)) #frame 리사이징 해서 넣음
    boxes, scores, classes, num = odapi.processFrame(frame) #검출기 가동############# num=총 검출 사물, person(class 1)이 아니어도 검출됨

    for i in range(len(boxes)): #박스 그리기

        if classes[i] == 1 and scores[i] > threshold: #1=person
            box = boxes[i]
            person += 1
            cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
            cv2.putText(frame, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
    
    #총 사람 수 프린트
    text = "{} : {}".format("person", person) 
    (H, W) = frame.shape[:2]
    cv2.putText(frame,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

    #사람 수 기록
    person_num[dic_path[path]]=person
    person_sum[dic_path[path]]+=person #1분째에 그 시간대 평균 인원수 기록용 합
    if MIrecording==True : #추가로 1분째면 기록
        person_avg=person_sum[dic_path[path]]/record_interval
        county[dic_path[path]][record_num]=person_avg #사람 수 셌던 기록
        person_sum[dic_path[path]]=0
        person_avg[dic_path[path]]=0

    return frame

def generate_frame(path):
    global person_num
    global person_sum
    global person_avg
    global county
    global record_interval
    global dic_path
    while True: #반복
        camera = cv2.VideoCapture(path)  # 카메라 장치 번호 (일반적으로 0), path로 설정해둠

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
                if frame_counter %interval ==0 : #1초마다 검출, 화면에 보여줌
                    if frame_counter % (record_interval * interval )==0 : #10초(1시간)마다 숫자 저장 예정
                        frame_counter += 1
                        record_num += 1
                        if(record_num>24):
                            record_num=0   
                        # frame=frame_processing(path,frame ,True ,record_num) #24시간 주기로 반복 (0~23)


                        frame = cv2.resize(frame, (800, 600)) #frame 리사이징 해서 넣음
                        boxes, scores, classes, num = odapi.processFrame(frame) #검출기 가동############# num=총 검출 사물, person(class 1)이 아니어도 검출됨
                        person = 0

                        for i in range(len(boxes)): #박스 그리기

                            if classes[i] == 1 and scores[i] > threshold: #1=person
                                box = boxes[i]
                                person += 1
                                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                                cv2.putText(frame, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                        
                        #총 사람 수 프린트
                        text = "{} : {}".format("person", person) 
                        (H, W) = frame.shape[:2]
                        cv2.putText(frame,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                        #사람 수 기록
                        person_num[dic_path[path]]=person
                        #append용, 60초되면 60으로 나눈 값을 person_avg에 넣을 거임
                        person_sum[dic_path[path]]+=person
#
                        #사람 수 기록
                        person_avg[dic_path[path]]=person_sum[dic_path[path]]/record_interval
                        county[dic_path[path]][record_num]=person_avg #사람 수 셌던 기록
                        person_sum[dic_path[path]]=0
                        person_avg[dic_path[path]]=0
 
 
                        # 프레임 스트리밍을 위해 웹으로 전송
                        ret, buffer = cv2.imencode('.jpg', frame)#('.jpg',frame,[압축품질])로 압축해서 출력속도 늘리기 아니면 이미지크기 갑 낮을 수록 압축높음
                        frame = buffer.tobytes() #문자열로 변환후 웹으로 전송
                        
                        # 프레임 반환
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

                        
                    else : #10초가 아니면 저장은 안해도 여전히 화면에 보여주긴함, 기록을 안하니 record_num 은 그대로
                        frame_counter += 1
                        # frame=frame_processing(path,frame,False,record_num)
                        person = 0
                        threshold = 0.7

                        frame = cv2.resize(frame, (800, 600)) #frame 리사이징 해서 넣음
                        boxes, scores, classes, num = odapi.processFrame(frame) #검출기 가동############# num=총 검출 사물, person(class 1)이 아니어도 검출됨

                        for i in range(len(boxes)): #박스 그리기

                            if classes[i] == 1 and scores[i] > threshold: #1=person
                                box = boxes[i]
                                person += 1
                                cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                                cv2.putText(frame, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                        
                        #총 사람 수 프린트
                        text = "{} : {}".format("person", person) 
                        (H, W) = frame.shape[:2]
                        cv2.putText(frame,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)


                        #사람 수 기록
                        person_num[dic_path[path]]=person
                        #append용, 60초되면 60으로 나눈 값을 person_avg에 넣을 거임
                        person_sum[dic_path[path]]+=person

                        # 프레임 스트리밍을 위해 웹으로 전송
                        ret, buffer = cv2.imencode('.jpg', frame)#('.jpg',frame,[압축품질])로 압축해서 출력속도 늘리기 아니면 이미지크기 갑 낮을 수록 압축높음
                        frame = buffer.tobytes() #문자열로 변환후 웹으로 전송
                        
                        # 프레임 반환
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')






            


@app.route('/') #기본 서버 페이지
def index():
    # 웹 페이지 템플릿 표시
    return render_template('indexfin.html')#해당 템플릿 호출


@app.route('/video_feed0')
def video_feed0():
    # 영상 스트리밍을 위한 Response 객체 반환
    return Response(generate_frame("./street_video0.MP4"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed1')
def video_feed1():
    # 영상 스트리밍을 위한 Response 객체 반환
    return Response(generate_frame("./street_video1.MP4"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed2')
def video_feed2():
    # 영상 스트리밍을 위한 Response 객체 반환
    return Response(generate_frame("./street_video2.MP4"), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed3')
def video_feed3():
    # 영상 스트리밍을 위한 Response 객체 반환
    return Response(generate_frame("./street_video3.MP4"), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/human_data0',methods=['GET'])
def get_person_num0():
    global person_num
    # 정수 데이터 생성
    data = person_num[0]
    # 데이터를 문자열로 변환하여 전송
    return str(data)

@app.route('/human_county0',methods=['GET'])
def get_county_num0():
    global county
    # 정수 데이터 생성
    data = county[0]
    #배열 전송가능 형태로 전환
    response = jsonify(data)
    # 배열 전송
    return response
@app.route('/human_data1',methods=['GET'])
def get_person_num1():
    global person_num
    global county
    # 정수 데이터 생성
    data = person_num[1]
    # 데이터를 문자열로 변환하여 전송
    return str(data)

@app.route('/human_county1',methods=['GET'])
def get_county_num1():
    global county
    # 정수 데이터 생성
    data = county[1]
    #배열 전송가능 형태로 전환
    response = jsonify(data)
    # 배열 전송
    return response
@app.route('/human_data2',methods=['GET'])
def get_person_num2():
    global person_num
    # 정수 데이터 생성
    data = person_num[2]
    # 데이터를 문자열로 변환하여 전송
    return str(data)

@app.route('/human_county2',methods=['GET'])
def get_county_num2():
    global county
    # 정수 데이터 생성
    data = county[2]
    #배열 전송가능 형태로 전환
    response = jsonify(data)
    # 배열 전송
    return response


if __name__ == '__main__':
    app.run(host='localhost', port=5000)

    while True:
        schedule.run_pending()
        time.sleep(0.1)

        #실시간으로 되긴하는데 작업복잡도때문에 텀이 좀 길다
        #컴퓨터 문제 일수도...