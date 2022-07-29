import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
import sys
import time
from process import Process
from webcam import Webcam
from video import Video
from interface import waitKey, plotXY
import math
import os
#from numba import jit

class Communicate(QObject):
    closeApp = pyqtSignal()
 

    
class GUI(QMainWindow, QThread):
    def __init__(self):
        
        super(GUI,self).__init__()
        self.initUI()
        self.webcam = Webcam()
        self.video = Video()
        self.input = self.webcam
        self.dirname = ""
        #print("Input: webcam")
        self.statusBar.showMessage("Input: webcam",1)
        self.btnOpen.setEnabled(False)
        self.process = Process()
        self.status = False
        self.frame = np.zeros((10,10,3),np.uint8)
        self.bpm = 0
        self.close_reason = False

    def initUI(self):
        
        font = QFont()
        font.setPointSize(16)
        
        font2 = QFont()
        font2.setPointSize(14)
        
        font3 = QFont()
        font3.setPointSize(15)
        
        font4 = QFont()
        font4.setPointSize(11)
        
        fontLight = QFont()
        fontLight.setPointSize(35)

#以下輸入用       
        #test_x = -450
        first_row = 550

        self.input_num = QLineEdit(self)
        self.input_num.move(550,first_row)
        self.input_num.setFixedWidth(185)
        self.input_num.setFixedHeight(45)
        self.input_num.setFont(font2)
        #self.input_num.setAlignment(Qt.AlignRight)
        self.input_num.setPlaceholderText("請輸入編號")
        
        self.input_num.setObjectName("input_num")
        
        
        self.cbb_gender = QComboBox(self)
        self.cbb_gender.addItem("請選擇生理性別")        
        self.cbb_gender.addItem("生理男")
        self.cbb_gender.addItem("生理女")        
        self.cbb_gender.setCurrentIndex(0)
        self.cbb_gender.setFixedWidth(170)
        self.cbb_gender.setFixedHeight(45)
        self.cbb_gender.move(760,first_row)
        self.cbb_gender.setFont(font2)

        
        
        self.input_age = QLineEdit(self)
        self.input_age.move(960,first_row)
        self.input_age.setFixedWidth(140)
        self.input_age.setFixedHeight(45)
        self.input_age.setFont(font2)
        
        self.input_age.setPlaceholderText("請輸入年齡")
        self.input_age.setObjectName("input_age")
        #self.input_age.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        
        second_row = 615
        
        self.input_height = QLineEdit(self)
        self.input_height.move(550,second_row)
        self.input_height.setFixedWidth(205)
        self.input_height.setFixedHeight(45)
        self.input_height.setFont(font2)
        
        self.input_height.setPlaceholderText("請輸入身高(cm)")
        self.input_height.setObjectName("input_height")
        #self.input_height.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        
        
        self.input_weight = QLineEdit(self)
        self.input_weight.move(785,second_row)
        self.input_weight.setFixedWidth(205)
        self.input_weight.setFixedHeight(45)
        self.input_weight.setFont(font2)
        
        self.input_weight.setPlaceholderText("請輸入體重(kg)")
        self.input_weight.setObjectName("input_weight")
        #self.input_weight.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        
        
        third_row = 680
        
        self.input_sitting_height = QLineEdit(self)
        self.input_sitting_height.move(550,third_row)
        self.input_sitting_height.setFixedWidth(205)
        self.input_sitting_height.setFixedHeight(45)
        self.input_sitting_height.setFont(font2)
        
        self.input_sitting_height.setPlaceholderText("請輸入坐高(cm)")
        #self.input_sitting_height.setObjectName("input_sitting_height")
        
        
        self.input_luminance = QLineEdit(self)
        self.input_luminance.move(785,third_row)
        self.input_luminance.setFixedWidth(170)
        self.input_luminance.setFixedHeight(45)
        self.input_luminance.setFont(font2)
        
        self.input_luminance.setPlaceholderText("請輸入光度")
        self.input_luminance.setObjectName("input_luminance")
        
        
        
        fourth_row = 870
        
        self.input_count = QLineEdit(self)
        self.input_count.move(630,fourth_row)
        self.input_count.setFixedWidth(1)
        self.input_count.setFixedHeight(1)
        self.input_count.setFont(font4)
        #self.input_count.setAlignment(Qt.AlignRight)
        self.input_count.setText("1000")
        self.input_count.setPlaceholderText("請勿隨意修改!")
        self.input_count.setObjectName("input_count")
        self.input_count.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        
        self.cbb_filename = QComboBox(self) 
        self.cbb_filename.addItem("請按Start")
        self.cbb_filename.addItem("依日期建檔")
        self.cbb_filename.addItem("依月份建檔")
        self.cbb_filename.addItem("依年份建檔")   
        self.cbb_filename.setCurrentIndex(0)
        self.cbb_filename.setFixedWidth(1)
        self.cbb_filename.setFixedHeight(1)
        self.cbb_filename.move(795,fourth_row)
        self.cbb_filename.setFont(font4)

        
        
        self.cbb_upload = QComboBox(self)        
        self.cbb_upload.addItem("不上傳雲端")
        #self.cbb_upload.addItem("上傳雲端")
        self.cbb_upload.setCurrentIndex(0)
        self.cbb_upload.setFixedWidth(1)
        self.cbb_upload.setFixedHeight(1)
        self.cbb_upload.move(960,fourth_row)
        self.cbb_upload.setFont(font4)
        '''
        column_first = 1500
        
        self.input_SBP = QLineEdit(self)
        self.input_SBP.move(column_first,780)
        self.input_SBP.setFixedWidth(200)
        self.input_SBP.setFixedHeight(45)
        self.input_SBP.setFont(font2)
        #self.input_luminance.setAlignment(Qt.AlignRight)
        self.input_SBP.setPlaceholderText("請輸入收縮壓")
        self.input_SBP.setObjectName("input_SBP")'''

#以上輸入用        

        #Start
        self.btnStart = QPushButton("Start", self)
        self.btnStart.move(515,750)
        self.btnStart.setFixedWidth(600)
        self.btnStart.setFixedHeight(100)
        self.btnStart.setFont(font)
        #信號與槽進行連接，信號可綁定普通成員函數
        self.btnStart.clicked.connect(self.run)
        
        #open
        self.btnOpen = QPushButton("Open", self)
        self.btnOpen.move(10,10)
        self.btnOpen.setFixedWidth(1)
        self.btnOpen.setFixedHeight(1)
        self.btnOpen.setFont(font)
        
        #信號與槽進行連接，信號可綁定普通成員函數
        self.btnOpen.clicked.connect(self.openFileDialog)
        
        self.cbbInput = QComboBox(self)
        #下拉列錶框組件類，它提供一個下拉列表供用戶選擇，也可以直接當作一個QLineEdit用作輸入。
        self.cbbInput.addItem("Webcam")
        self.cbbInput.setCurrentIndex(0)
        self.cbbInput.setFixedWidth(1)
        self.cbbInput.setFixedHeight(1)
        self.cbbInput.move(10,10)
        self.cbbInput.setFont(font)
        self.cbbInput.activated.connect(self.selectInput)
        
        self.lblDisplay = QLabel(self) 
        self.lblDisplay.setGeometry(500,10,675,525)#(530,10,720,560)
        self.lblDisplay.setStyleSheet("background-color: #000000")
        
        self.lblROI = QLabel(self) 
        self.lblROI.setGeometry(1200,10,200,200)#(1290,10,260,260)
        self.lblROI.setStyleSheet("background-color: #000000")

        

#以下數據
        '''self.lblHR = QLabel(self)
        self.lblHR.setGeometry(1290,310,500,80)
        self.lblHR.setFont(font)
        self.lblHR.setText("Heart rate1: ")'''
        
        geometry_statistics_x=1180
        
        
        self.lblHR2 = QLabel(self)
        self.lblHR2.setGeometry(geometry_statistics_x,220,500,80)
        self.lblHR2.setFont(font)
        self.lblHR2.setText("Heart rate: ")
        
        self.lblHF = QLabel(self)
        #label to show stable HF
        self.lblHF.setGeometry(geometry_statistics_x,300,400,80)
        self.lblHF.setFont(font)
        self.lblHF.setText("HF(nu): ")
        
        self.lblLF = QLabel(self)
        #label to show stable LF
        self.lblLF.setGeometry(geometry_statistics_x,380,400,80)
        self.lblLF.setFont(font)
        self.lblLF.setText("LF(nu): ")
        
        self.lblLFHF = QLabel(self)
        #label to show stable LF/HF
        self.lblLFHF.setGeometry(geometry_statistics_x,460,400,80)
        self.lblLFHF.setFont(font)
        self.lblLFHF.setText("LF/HF: ")
        
        self.lbldLFHF = QLabel(self)
        #label to show stable dLF/HF
        self.lbldLFHF.setGeometry(geometry_statistics_x,540,400,80)
        self.lbldLFHF.setFont(font)
        self.lbldLFHF.setText("d(LF/HF)/dt: ")
        
        #self.lblBlinks = QLabel(self)
        #self.lblBlinks.setGeometry(geometry_statistics_x,620,400,80)
        #self.lblBlinks.setFont(font)
        #self.lblBlinks.setText("Blinks: ")
        
        self.lblBP = QLabel(self)
        #label to show BloodPresure
        self.lblBP.setGeometry(geometry_statistics_x,620,400,80)
        self.lblBP.setFont(font)
        #self.lblBP.setText("Blood Pressure: ")
        
        
        self.lbltotaltime = QLabel(self)
        self.lbltotaltime.setGeometry(geometry_statistics_x,700,400,80)
        self.lbltotaltime.setFont(font)
        self.lbltotaltime.setText("Total Time: ")
        
        self.lblfatigue = QLabel(self)
        self.lblfatigue.setGeometry(geometry_statistics_x,780,400,80)
        self.lblfatigue.setFont(font)
        self.lblfatigue.setText("疲勞等級: ")
        
#以上數據   
        
#以下動圖標題
        
        self.titleL1 = QLabel(self)
        self.titleL1.setGeometry(15,1,400,30)
        self.titleL1.setFont(font3)
        self.titleL1.setText("rPPG: ")
        
        #self.titleL2 = QLabel(self)
        #self.titleL2.setGeometry(15,224,400,30)
        #self.titleL2.setFont(font3)
        #self.titleL2.setText("rPPG_G: ")
        
        self.titleL3 = QLabel(self)
        self.titleL3.setGeometry(15,224,400,30)
        self.titleL3.setFont(font3)
        self.titleL3.setText("FFT_HR: ")
        
        self.titleL4 = QLabel(self)
        self.titleL4.setGeometry(15,447,400,30)
        self.titleL4.setFont(font3)
        self.titleL4.setText("FFT_HRV: ")
        
        self.titleR1 = QLabel(self)
        self.titleR1.setGeometry(1425,1,400,30)
        self.titleR1.setFont(font3)
        self.titleR1.setText("LF&HF&LF/HF: ")
        
        self.titleR2 = QLabel(self)
        self.titleR2.setGeometry(1425,224,400,30)
        self.titleR2.setFont(font3)
        self.titleR2.setText("SampEn: ")
        
        self.titleR3 = QLabel(self)
        self.titleR3.setGeometry(1425,447,400,30)
        self.titleR3.setFont(font3)
        self.titleR3.setText("ApEn: ")
        
        
        '''
        self.titleR4 = QLabel(self)
        self.titleR4.setGeometry(1425,670,400,30)
        self.titleR4.setFont(font)
        self.titleR4.setText(" ")
        '''
        
#以上動圖標題       

#以下動圖
        
        self.signal_PPG_Plt = pg.PlotWidget(self)        
        self.signal_PPG_Plt.move(10,31)
        self.signal_PPG_Plt.resize(470,192)
        self.signal_PPG_Plt.setBackground('#E0E0E0') 
        self.signal_PPG_Plt.setLabel('bottom', "rPPG") 
        
        
        #self.signal_Plt = pg.PlotWidget(self)        
        #elf.signal_Plt.move(10,254)
        #self.signal_Plt.resize(470,192)#(470,192)
        #self.signal_Plt.setBackground('#E0E0E0') 
        #self.signal_Plt.setLabel('bottom', "rPPG_G") 
        
        
        #self.signal_PPGcompare_Plt = pg.PlotWidget(self)        
        #self.signal_PPGcompare_Plt.move(10,477)
        #self.signal_PPGcompare_Plt.resize(470,192)
        #self.signal_PPGcompare_Plt.setBackground('#E0E0E0') 
        #self.signal_PPGcompare_Plt.setLabel('bottom', "rPPG_Compare") 
        
        
        self.fft_Plt = pg.PlotWidget(self)
        self.fft_Plt.setBackground('#E0E0E0')        
        self.fft_Plt.move(10,254)
        self.fft_Plt.resize(470,192)
        self.fft_Plt.setLabel('bottom', "FFT") 
        
        
        self.fft2_Plt = pg.PlotWidget(self)
        self.fft2_Plt.setBackground('#E0E0E0')
        self.fft2_Plt.move(10,477)
        self.fft2_Plt.resize(470,192)        
        self.fft2_Plt.setRange(xRange = (0,0.5))
        self.fft2_Plt.setLabel('bottom', "FFT_HRV") 
        
        
        self.hrv_Plt = pg.PlotWidget(self)
        self.hrv_Plt.setBackground('#E0E0E0')
        self.hrv_Plt.move(1420,31)
        self.hrv_Plt.resize(470,192)        
        self.hrv_Plt.setLabel('bottom', "LF(nu) & HF(nu)") 
        
        self.SampEn_Plt = pg.PlotWidget(self)
        self.SampEn_Plt.setBackground('#E0E0E0')
        self.SampEn_Plt.move(1420,254)
        self.SampEn_Plt.resize(470,192)        
        self.SampEn_Plt.setLabel('bottom', "SampEn") 
        
        self.ApEn_Plt = pg.PlotWidget(self)
        self.ApEn_Plt.setBackground('#E0E0E0')
        self.ApEn_Plt.move(1420,477)
        self.ApEn_Plt.resize(470,192)        
        self.ApEn_Plt.setLabel('bottom', "ApEn") 
        
        '''self.fft3_Plt.move(10,640)
        self.fft3_Plt.resize(480,192)
        self.fft3_Plt.setLabel('bottom', "FFT2_show") '''
        
        '''self.bp_Plt1 = pg.PlotWidget(self)
        
        self.bp_Plt1.move(10,430)
        self.bp_Plt1.resize(480,192)
        self.bp_Plt1.setStyleSheet("background-color: #F1F1F1")
        self.bp_Plt1.setLabel('bottom', "PPG")'''
        
        
        '''self.bp_Plt2 = pg.PlotWidget(self)
        
        self.bp_Plt2.move(10,640)
        self.bp_Plt2.resize(480,192)
        self.bp_Plt2.setStyleSheet("background-color: #F1F1F1")
        self.bp_Plt2.setLabel('bottom', "一階")'''
        
        
        
        
        
        self.statusBar = QStatusBar()
        self.statusBar.setFont(font)
        self.setStatusBar(self.statusBar)
        
        self.c = Communicate()
        self.c.closeApp.connect(self.close)
        
        self.setGeometry(10,100,1900,950)
        self.setStyleSheet("background-color:#CCD8CF")
        self.setWindowTitle("可攜式非接觸即時疲勞偵測系統")
        
        self.show()
#以上動圖        

#以下提醒實驗人員
        lblWarning1 = " ~~~給實驗人員的小提醒~~~\n"
        lblWarning2 = "  *1.檢查資料抓幾筆\n"
        lblWarning3 = "  *2.注意雲端和Excel是否關閉\n"
        lblWarning4 = "  *3.幫忙測量資料並協助輸入\n"
        lblWarning5 = "  *4.建檔選項需注意是否選擇正確"
        lblWarning = lblWarning1+lblWarning2+lblWarning3+\
                                lblWarning4+lblWarning5        
        
        #QboxWarning = QMessageBox.warning(self,"~小提醒~",lblWarning,QMessageBox.Ok, QMessageBox.Ok)
        
#以上提醒實驗人員           
        
        
        
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(300)

        
        
        
        
        
    def update(self):
        
        
        self.signal_PPG_Plt.clear()
        self.signal_PPG_Plt.plot(self.process.samples[20:],pen=pg.mkPen('#000000',width=2))
        

        #self.signal_Plt.clear()
        #self.signal_Plt.plot(self.process.samples[20:],pen=pg.mkPen('#00A600',width=2))
        
        
        #self.signal_PPGcompare_Plt.clear()
        #self.signal_PPGcompare_Plt.plot(self.process.samplesPPG[20:],pen=pg.mkPen('#000000',width=2))
        #self.signal_PPGcompare_Plt.plot(self.process.samples[20:],pen=pg.mkPen('#00A600',width=2))
        
        
        
        
        
        
        self.fft_Plt.clear()
        self.fft_Plt.plot(np.column_stack((self.process.freqs, self.process.fft)), pen=pg.mkPen('#FF0000',width=2))
        self.line_hr = pg.InfiniteLine(angle=90, movable=True,pen=pg.mkPen('#750000',width=3)) 
        self.fft_Plt.addItem(self.line_hr) 
        self.line_hr.setBounds([self.process.bpm,self.process.bpm])
       
        
        
        #列合併：np.column_stack()
        self.fft2_Plt.clear()
        self.fft2_Plt.plot(np.column_stack((self.process.freqs2, self.process.fft2)), pen=pg.mkPen('#FF2D2D',width=2))
        self.lr_LF=pg.LinearRegionItem([0.053,0.15], bounds=[0.053,0.15], movable=False,brush=(192,255,62,100))         
        self.fft2_Plt.addItem(self.lr_LF) 
        self.lr_HF=pg.LinearRegionItem([0.15,0.4], bounds=[0.15,0.4], movable=False,brush=(135,206,235,100)) 
        self.fft2_Plt.addItem(self.lr_HF) 
        
        
        self.hrv_Plt.clear()
        self.hrv_Plt.plot(self.process.lf_array, pen = pg.mkPen('#0000C6',width=2))
        self.hrv_Plt.plot(self.process.hf_array, pen = pg.mkPen('#FF0000',width=2))
        self.hrv_Plt.plot(self.process.lfhf_array, pen = pg.mkPen('#006000',width=2))

        self.SampEn_Plt.clear()
        self.SampEn_Plt.plot(self.process.SampEn_data, pen = pg.mkPen('#8F4586',width=2))
        
        self.ApEn_Plt.clear()
        self.ApEn_Plt.plot(self.process.ApEn_data, pen = pg.mkPen('#00A8A8',width=2))
        
        '''self.fft3_Plt.clear()
        self.fft3_Plt.plot(np.column_stack((self.process.freqs3, self.process.fft3)), pen = 'g')'''
        #self.bp_Plt1.clear()
        #self.bp_Plt1.plot(self.process.now_time,self.process.bp, pen = 'g')
        
        # self.x_derivative1 
        #self.bp_Plt2.clear()
        #self.bp_Plt2.plot(self.process.x_derivative1,pen='g')

        
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def closeEvent(self, event):
        if self.close_reason == True:

            reply = QMessageBox.critical(self,"資料錯誤 !", "提醒您 :\n請重新開啟程式，\n並請在開始偵測前，\n確認資料是否填寫完整~~~",
                QMessageBox.Ok, QMessageBox.Ok)
        else :
            reply = QMessageBox.information(self,"Message", "掰掰 !",
                QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes or reply == QMessageBox.Ok :

            event.accept()
            self.input.stop()
            cv2.destroyAllWindows()
        else: 
            event.ignore()    
        
        

    def on_closing(self, event):
        self.c.closeApp.emit()    
    
    def key_handler(self):
        self.pressed = waitKey(1) & 255  
        if self.pressed == 27:  
            print("[INFO] Exiting")
            self.webcam.stop()
            sys.exit()
    
    def openFileDialog(self):
        self.dirname = QFileDialog.getOpenFileName(self, 'OpenFile',r"C:\Users\uidh2238\Desktop\test videos")

    def selectInput(self):
        self.reset()
        if self.cbbInput.currentIndex() == 0:
            self.input = self.webcam
            print("Input: webcam")
            self.btnOpen.setEnabled(False)
        elif self.cbbInput.currentIndex() == 1:
            self.input = self.video
            print("Input: video")
            self.btnOpen.setEnabled(True)
    

            
    def reset(self):
        self.process.reset()
        self.lblDisplay.clear()
        self.lblDisplay.setStyleSheet("background-color: #000000")

    @QtCore.pyqtSlot()
    def main_loop(self):
        
        
            
        frame = self.input.get_frame()
        self.process.frame_in = frame
        self.process.run()
   
        self.frame = self.process.frame_out 
        self.f_fr = self.process.frame_ROI 

        self.bpm = self.process.bpm 
        
        
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR)
        self.process.GUIframe = self.frame
        cv2.putText(self.frame, "FPS "+str(float("{:.2f}".format(self.process.fps))),
                       (20,460), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255),2)
        
            
            
        
        img = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], 
                        self.frame.strides[0], QImage.Format_RGB888)
        self.lblDisplay.setPixmap(QPixmap.fromImage(img))
        
        
        self.f_fr = cv2.cvtColor(self.f_fr, cv2.COLOR_RGB2BGR)
        self.f_fr = np.transpose(self.f_fr,(0,1,2)).copy()
        f_img = QImage(self.f_fr, self.f_fr.shape[1], self.f_fr.shape[0], 
                       self.f_fr.strides[0], QImage.Format_RGB888)
        self.lblROI.setPixmap(QPixmap.fromImage(f_img))
        
        
        ##後面跑數字
        #self.lblBlinks.setText("Blinks: " + "\n"+ " "+str(self.process.blinks)+" times")
        if self.process.ann_predict >=0:
            self.lblfatigue.setText("疲勞等級: " + "\n"+ " "+str(self.process.ann_predict)+" 級")
        self.lbltotaltime.setText("Total Time: \n"+("{:.2f}".format((self.process.progress)))+" %")
        if self.process.bpms.__len__() >50:
            if(max(self.process.bpms-np.mean(self.process.bpms))<2):
                self.GUI_bpms=math.trunc(round(np.mean(self.process.bpms)))
                self.lblHR2.setText("Heart rate: " + "\n"+ " "+str(self.GUI_bpms) + " bpm")
                self.process.GUI_bpms=self.GUI_bpms
                #self.lblBP.setText("Blood Pressure: " + "\n"+ " "+("{:.2f}".format((self.process.SBP_DBP))) +'/'+ ("{:.2f}".format((self.process.SBP_DBP2))))
                
                
        if self.process.lfhf>0:
            self.lblLF.setText("LF(nu): " + "\n"+ " "+str(float("{:.2f}".format(np.mean(self.process.lf)))) + " ")
            self.lblHF.setText("HF(nu): " + "\n"+ " "+str(float("{:.2f}".format(np.mean(self.process.hf)))) + " ")
            self.lblLFHF.setText("LF/HF: " + "\n"+ " "+str(float("{:.2f}".format(np.mean(self.process.lfhf)))) + " ")
            self.lbldLFHF.setText("d(LF/HF)/dt: " + "\n"+ " "+str(float("{:.2f}".format(self.process.dlfhf))) + " ")
            #self.lbltotaltime.setText("Total Time: " + "\n"+ " "+str(float("{:.2f}".format(self.process.total_time))) + " (s)")
            
        #if self.process.file==False and self.process.progress>=99.99:
            #self.lbltotaltime.setText("Total Time:\n正在儲存... ")
            
        if self.process.file==True:            
            #self.lbltotaltime.setText("Total Time: " + "\n"+str(float("{:.2f}".format(self.process.total_time))) + " (s) "+ str(float("{:.2f}".format(self.process.total_time_file)))+ " (s)")
            self.lbltotaltime.setText("Total Time:" + str(float("{:.2f}".format(self.process.total_time_file)))+ "s\n量測完畢!感謝您!!! ") 
        
        
            
        self.key_handler()  

    def run(self, input):
        
        self.reset()
        #以下輸入
        self.subject_num = self.input_num.text()
        self.process.subject_num=self.subject_num
        
        self.subject_age = self.input_age.text()
        self.process.subject_age=self.subject_age

        self.subject_height = self.input_height.text()
        self.process.subject_height=self.subject_height
        
        self.subject_weight = self.input_weight.text()
        self.process.subject_weight=self.subject_weight
        
        self.subject_sitting_height = self.input_sitting_height.text()
        self.process.subject_sitting_height=self.subject_sitting_height
        
        self.subject_luminance = self.input_luminance.text()
        self.process.subject_luminance=self.subject_luminance
        
        self.subject_count = self.input_count.text()
        self.process.subject_count=self.subject_count
        
        if self.cbb_gender.currentIndex() == 1:
            self.subject_gender = "男"
            self.process.subject_gender=self.subject_gender
            
            
        elif self.cbb_gender.currentIndex() == 2:
            self.subject_gender = "女"
            self.process.subject_gender=self.subject_gender
        
        elif self.cbb_gender.currentIndex() == 0:
            self.subject_gender = "unknown"
            self.process.subject_gender=self.subject_gender
        
        
        if self.cbb_filename.currentIndex() == 0:
            self.subject_filename = 0
            self.process.subject_filename=self.subject_filename
            
        elif self.cbb_filename.currentIndex() == 1:
            self.subject_filename = 1
            self.process.subject_filename=self.subject_filename
            
        elif self.cbb_filename.currentIndex() == 2:
            self.subject_filename = 2
            self.process.subject_filename=self.subject_filename
        
        if self.cbb_upload.currentIndex() == 0:
            
            self.process.subject_upload=False
            
        elif self.cbb_upload.currentIndex() == 1:
            
            self.process.subject_upload=True
            
        if not self.subject_age or not self.subject_num \
                or not self.subject_height or not self.subject_weight\
                or not self.subject_sitting_height\
                or not self.subject_luminance\
                or not self.subject_count: 
            self.close_reason = True
            self.c.closeApp.emit()
        #以上輸入
        if self.close_reason == False:
            input = self.input
            self.input.dirname = self.dirname
            if self.input.dirname == "" and self.input == self.video: 
                print("choose a video first")
                return
            if self.status == False:
                self.status = True
                input.start()
                self.btnStart.setText("Stop")
                self.cbbInput.setEnabled(False)
                self.btnOpen.setEnabled(False)
                
                while self.status == True:
                    self.main_loop()
            elif self.status == True:
                self.status = False
                input.stop()
                self.btnStart.setText("Start")
                self.cbbInput.setEnabled(True)
            
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GUI()
    while ex.status == True:
        ex.main_loop()

    sys.exit(app.exec_())

   


