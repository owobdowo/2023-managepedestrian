# # imported necessary library
# from tkinter import *
# import tkinter as tk # GUI구성
# import tkinter.messagebox as mbox #메세지 창
# from tkinter import filedialog #저장 파일 관련
from PIL import ImageTk, Image #이미지 처리, 가공
import cv2
import argparse #파이썬 파일 실행시 옵션을 줄 수 있도록 하는 
from persondetection import DetectorAPI #이 파일에 판정 코드 들어있음
# import matplotlib.pyplot as plt #통계치 표시
# from fpdf import FPDF #pdf파일제작

import flask
from flask import request
from flask import Response
from flask import stream_with_context

def argsParser():
        arg_parse = argparse.ArgumentParser()
        # arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
        # arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
        arg_parse.add_argument("-c", "--camera", default=True, help="Set true if you want to use the camera.")
        arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file") #output입력시 해당경로로 비디오 파일 저장
        args = vars(arg_parse.parse_args())
        return args

    # ---------------------------- camera section ------------------------------------------------------------
def camera_option():
        # new window created for camera section
        # windowc = tk.Tk()
        # windowc.title("Human Detection from Camera")
        # windowc.iconbitmap('Images/icon.ico')
        # windowc.geometry('1000x700')

        max_count3 = 0
        framex3 = []
        county3 = []
        max3 = []
        avg_acc3_list = []
        max_avg_acc3_list = []
        max_acc3 = 0
        max_avg_acc3 = 0
        # function defined to open the camera
        def open_cam(): #버튼 활성화 됐을 시 이런 저런 거 검토하고 if True 조건으로 가서 실행
            global max_count3, framex3, county3, max3, avg_acc3_list, max_avg_acc3_list, max_acc3, max_avg_acc3
            max_count3 = 0
            framex3 = []
            county3 = []
            max3 = []
            avg_acc3_list = []
            max_avg_acc3_list = []
            max_acc3 = 0
            max_avg_acc3 = 0

            args = argsParser()

            # info1.config(text="Status : Opening Camera...")
            # info2.config(text="                                                  ")
            # mbox.showinfo("Status", "Opening Camera...Please Wait...", parent=windowc)
            # time.sleep(1)

            writer = None
            if args['output'] is not None:
                writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))#동영상 저장(저장형식,프레임,크기)
            if True:
                detectByCamera(writer)

        # function defined to detect from camera
        def detectByCamera(writer): #관련 기능 시작
            #global variable created
            global max_count3, framex3, county3, max3, avg_acc3_list, max_avg_acc3_list, max_acc3, max_avg_acc3
            max_count3 = 0
            framex3 = []
            county3 = []
            max3 = []
            avg_acc3_list = []
            max_avg_acc3_list = []
            max_acc3 = 0
            max_avg_acc3 = 0

            # function defined to plot the people count in camera
            # 수치 관련 표 제작
            # def cam_enumeration_plot():
            #     plt.figure(facecolor='orange', )
            #     ax = plt.axes()
            #     ax.set_facecolor("yellow")
            #     plt.plot(framex3, county3, label="Human Count", color="green", marker='o', markerfacecolor='blue')
            #     plt.plot(framex3, max3, label="Max. Human Count", linestyle='dashed', color='fuchsia')
            #     plt.xlabel('Time (sec)')
            #     plt.ylabel('Human Count')
            #     plt.legend()
            #     plt.title("Enumeration Plot")
            #     plt.get_current_fig_manager().canvas.set_window_title("Plot for Camera")
            #     plt.show()

            # def cam_accuracy_plot():
            #     plt.figure(facecolor='orange', )
            #     ax = plt.axes()
            #     ax.set_facecolor("yellow")
            #     plt.plot(framex3, avg_acc3_list, label="Avg. Accuracy", color="green", marker='o', markerfacecolor='blue')
            #     plt.plot(framex3, max_avg_acc3_list, label="Max. Avg. Accuracy", linestyle='dashed', color='fuchsia')
            #     plt.xlabel('Time (sec)')
            #     plt.ylabel('Avg. Accuracy')
            #     plt.title('Avg. Accuracy Plot')
            #     plt.legend()
            #     plt.get_current_fig_manager().canvas.set_window_title("Plot for Camera")
            #     plt.show()

            # def cam_gen_report():
            #     pdf = FPDF(orientation='P', unit='mm', format='A4')
            #     pdf.add_page()
            #     pdf.set_font("Arial", "", 20)
            #     pdf.set_text_color(128, 0, 0)
            #     pdf.image('Images/Crowd_Report.png', x=0, y=0, w=210, h=297)

            #     pdf.text(125, 150, str(max_count3))
            #     pdf.text(105, 163, str(max_acc3))
            #     pdf.text(125, 175, str(max_avg_acc3))
            #     if (max_count3 > 25):
            #         pdf.text(26, 220, "Max. Human Detected is greater than MAX LIMIT.")
            #         pdf.text(70, 235, "Region is Crowded.")
            #     else:
            #         pdf.text(26, 220, "Max. Human Detected is in range of MAX LIMIT.")
            #         pdf.text(65, 235, "Region is not Crowded.")

            #     pdf.output('Crowd_Report.pdf')
            #     mbox.showinfo("Status", "Report Generated and Saved Successfully.", parent = windowc)
            #카메라 연결
            video = cv2.VideoCapture(0)
            odapi = DetectorAPI()
            threshold = 0.7

            x3 = 0
            while True: #출력
                check, frame = video.read() #비디오 설정
                img = cv2.resize(frame, (800, 600)) #이미지에 비디오 프레임별로 넣음
                boxes, scores, classes, num = odapi.processFrame(img) #검출기 가동
                person = 0
                acc = 0
                for i in range(len(boxes)): #박스 그리기

                    if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        person += 1
                        cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)  # cv2.FILLED
                        cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)  # (75,0,130),
                        acc += scores[i]
                        if (scores[i] > max_acc3):
                            max_acc3 = scores[i]
                text = "{} : {}".format("person", person)
                (H, W) = frame.shape[:2]
                cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255),2)

                if (person > max_count3): #그냥 정도 카운트
                    max_count3 = person

                if writer is not None:
                    writer.write(img)

                cv2.imshow("Human Detection from Camera", img) #imshow
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

                county3.append(person)
                x3 += 1
                framex3.append(x3)
                #사람 수 계산
                if(person>=1):
                    avg_acc3_list.append(acc / person)
                    if ((acc / person) > max_avg_acc3):
                        max_avg_acc3 = (acc / person)
                else:
                    avg_acc3_list.append(acc)

            video.release()
            # info1.config(text="                                                  ")
            # # info2.config(text="                                                  ")
            # info1.config(text="Status : Detection & Counting Completed")
            # info2.config(text="Max. Human Count : " + str(max_count3))
            cv2.destroyAllWindows()
#숫자 카운트
            for i in range(len(framex3)):
                max3.append(max_count3)
                max_avg_acc3_list.append(max_avg_acc3)

            # Button(windowc, text="Enumeration\nPlot", command=cam_enumeration_plot, cursor="hand2", font=("Arial", 20),bg="orange", fg="blue").place(x=100, y=530)
            # Button(windowc, text="Avg. Accuracy\nPlot", command=cam_accuracy_plot, cursor="hand2", font=("Arial", 20),bg="orange", fg="blue").place(x=700, y=530)
            # Button(windowc, text="Generate  Crowd  Report", command=cam_gen_report, cursor="hand2", font=("Arial", 20),bg="gray", fg="blue").place(x=325, y=550)
        
        
        if True:
            detectByCamera(None)
        #윈도우 관련, 필요없음
        {
        # lbl1 = tk.Label(windowc, text="DETECT  FROM\nCAMERA", font=("Arial", 50, "underline"), fg="brown")  # same way bg
        # lbl1.place(x=230, y=20)

        # Button(windowc, text="OPEN CAMERA", command=open_cam, cursor="hand2", font=("Arial", 20), bg="light green", fg="blue").place(x=370, y=230)

        # info1 = tk.Label(windowc, font=("Arial", 30), fg="gray")  # same way bg
        # info1.place(x=100, y=330)
        # info2 = tk.Label(windowc, font=("Arial", 30), fg="gray")  # same way bg
        # info2.place(x=100, y=390)

        # # function defined to exit from the camera window
        # def exit_winc():
        #     if mbox.askokcancel("Exit", "Do you want to exit?", parent = windowc):
        #         windowc.destroy()
        # windowc.protocol("WM_DELETE_WINDOW", exit_winc)
        }
camera_option()