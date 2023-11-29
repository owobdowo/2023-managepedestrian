import numpy as np
import tensorflow as tf
# import cv2
import time
import os
# import tensorflow.compat.v1 as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


class DetectorAPI:
    def __init__(self): #self를 사용하는 이유 function()으로 나둬도 인스턴스면 항상 그 객체 자체가 변수로 전송됨 
        #ex c=class() c.self_function(self) class.nonself_function()이 맞고 c.nonself(self)<변수 필요x class.self() 변수 필요라 trace back 오류
        path = os.path.dirname(os.path.realpath(__file__))#파일 받을 이름
        self.path_to_ckpt = f'frozen_inference_graph.pb'#경로 설정, 학습된 모델
        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)
#해당인스턴스안에 있는 변수에 detection_graph에서 나오는 값들을 넣어줌
#get_tensor_bt_name : 변수이름으로 값 호출
        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
#이걸로 값들 다 호출한 후 돌려주고 메인 파일에서 활용함(표시좌표는 이쪽에서 계산)

    def processFrame(self, image): #훈련 모델 깊이 확장? 출력화면 표시?
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()#시간
        #클래스 종류
        #인간 박스, 정확도 점수, 클래스, 숫자
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores,
                self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        #시간
        end_time = time.time()

        # print("Elapsed Time:", end_time-start_time)
        # print(self.image_tensor, image_np_expanded)
        im_height, im_width, _ = image.shape #이미지 가로 세로
        boxes_list = [None for i in range(boxes.shape[1])] #측정된 박스리스트 정의
        #없거나 i만큼 반복해서 삽입될듯
        for i in range(boxes.shape[1]): #박스 크기 좌우상하 꼭짓점 지정? i번쨰 박스의 박스리스트 값 ==박스리스트는 2차원배열
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),int(boxes[0, i, 1]*im_width),int(boxes[0, i, 2] * im_height),int(boxes[0, i, 3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()