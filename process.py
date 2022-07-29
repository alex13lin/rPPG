# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 14:12:03 2021

@author: a1016
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import statistics
import random
import time
from face_detection import FaceDetection
from sklearn.decomposition import PCA
from scipy import signal
# from sklearn.decomposition import FastICA
import pandas as pd
import pandas
import time
from bp_calculation import BP
import heapq
import math
import os
import zipfile
import datetime

from scipy.spatial import distance as dist
from openpyxl import Workbook,load_workbook
import os

import saving
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

class Process(object):
    
    def __init__(self):
        self.now_time=[]
        self.now_time2=[]
        self.now_time3=[]
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.samplesPPG = []
        self.buffer_size = 100
        self.buffer_size2 = 1200
        self.buffer_sizePPG = 100
        self.times = [] 
        self.times2 = []   
        self.timesPPG = []
        self.data_buffer = []
        self.data_buffer2 = []
        self.data_bufferPPG = []
        self.time_buffer=[]
        self.time_buffer2=[]
        self.bp_buffer = []
        self.fps = 0
        self.fpsPPG = 0
        self.fft = []
        self.fft2 = []
        self.freqs = []
        self.freqs2 = []
        self.t0 = time.time()   
        self.t0PPG = time.time()
        self.bpm = 0
        self.fd = FaceDetection()
        self.bpms = []
        self.peaks = []
        self.bp=[]
        self.max=[]
        self.min=[]
        self.x_derivative1=[]
        self.x_derivative2=[]
        self.BP=BP()
        self.SBP_DBP=0
        self.SBP_DBP2=0
        self.bpolist=[]
        self.now_timeoliset=[]
        self.hf=0
        self.lf=0
        self.hf_area=0
        self.lf_area=0
        self.lfhf=0
        self.dlfhf=0
        self.hfnu=[]
        self.lfnu=[]
        self.lfhfnu=[]
        self.hf_array=[]
        self.lf_array=[]
        self.lfhf_array=[]
        self.dlfhf_array=[]
        self.hf_data=[]
        self.lf_data=[]
        self.hf_area_data=[]
        self.lf_area_data=[]
        self.lfhf_data=[]
        self.dlfhf_data=[]         
        self.count=0
        self.last_hflf=1
        self.fatigue_time1=0
        self.fatigue_time2=0
        #self.start_time=time.time()
        self.start_time_status=0
        self.file=False
        self.time_array=[]
        self.SampEn=0
        self.SampEn_data=[]
        self.mode_array=[]
        self.hr=0
        self.SampEn_data=[]
        self.ApEn_data=[]
        self.blinks_counter=0
        self.blinks=0
        self.file_saving = False
        self.progress = 0
        self.SBP_DBP_data=[]
        self.SBP_DBP2_data=[]
        self.theANN_result_array=[]
        self.theANN_result=[]
        self.ann_predict = -1
        
    def Z_ScoreNormalization(self,x,mu,sigma):
        x = (x - mu) / sigma;
        return x;
    def smoothTriangle(self,data, degree=25):
        triangle=np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1])) # up then down
        smoothed=[]
    
        for i in range(degree, len(data) - degree * 2):
            point=data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point)/np.sum(triangle))
        # Handle boundaries
        smoothed=[smoothed[0]]*int(degree + degree/2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed       
    def get_derivative1(self,x,maxiter=10):   #//默認使用牛頓法迭代10次
        h=0.0001    
        #f=lambda x: x**2-c
        #f=lambda x: x**2-2*x-4
        F=lambda x: 0.5*x**2*(4-x)
        def df(x,f=F):                      #//使用導數定義法求解導數
                return (f(x+h)-f(x))/h

        for i in range(maxiter):
                x=x-F(x)/df(x)   #//計算一階導數，即是求解  f(x)=0  
                #x=x-df(x)/df(x,df) # //計算二階導數，即是求解  f'(x)=0
                #print (i+1,x)
        return x
    
    def get_derivative2(self,x,maxiter=10):   #//默認使用牛頓法迭代10次
        h=0.0001    
        #f=lambda x: x**2-c
        #f=lambda x: x**2-2*x-4
        F=lambda x: 0.5*x**2*(4-x)
        def df(x,f=F):                      #//使用導數定義法求解導數
                return (f(x+h)-f(x))/h

        for i in range(maxiter):
                #x=x-F(x)/df(x)   #//計算一階導數，即是求解  f(x)=0  
                x=x-df(x)/df(x,df) # //計算二階導數，即是求解  f'(x)=0
                #print (i+1,x)
        return x
    

    def ICA(self,norm):
        X=(norm,[1*random.random() for i in range(len(norm))])
        ICA = FastICA(n_components=1,max_iter=1000,tol=1e-6)
        X_transformed = ICA.fit_transform(X)
        A_ =  ICA.mixing_.T

        return A_ 
    def extractColor(self, frame):
        r = np.mean(frame[:,:,0])
        g = np.mean(frame[:,:,1])
        b = np.mean(frame[:,:,2])
        ppg=(r+g+b)/3

        return g,ppg
    
    def run(self):
        if self.start_time_status==0:
            self.start_time=time.time()
            self.start_time_status=1
        frame, face_frame, ROI1, ROI2, status, mask , leftEye, rightEye = self.fd.face_detect(self.frame_in)
        self.frame_out = frame
        self.frame_ROI = face_frame       
        g1,ppg1 = self.extractColor(ROI1)
        g2,ppg2 = self.extractColor(ROI2)
        
        L = len(self.data_buffer)
        N = len(self.data_buffer2)
        #P = len(self.data_bufferPPG)
        
        g = (g1+g2)/2 
        g_f=g
        #ppg=(ppg1+ppg2)/2
        
        self.check = False
        
        if(abs(g-np.mean(self.data_buffer))>10 and L>99):
            g = self.data_buffer[-1]
            
        if(abs(g_f-np.mean(self.data_buffer2))>10 and N>499):
            g_f = self.data_buffer2[-1]
            
            
        if g!=0:
            self.times.append(time.time() - self.t0)
            self.data_buffer.append(g)
        
        ##加入時間
            if len(self.time_buffer)>0:
                self.time_buffer.append(abs(round(time.time()-self.time_buffer[0],4)))
            else:
                self.time_buffer.append(time.time()) 
            if len(self.time_buffer)>301:
                self.time_buffer.pop(1)
            if L > self.buffer_size:
                    self.data_buffer = self.data_buffer[-self.buffer_size:]
                    self.times = self.times[-self.buffer_size:]
                    self.bpms = self.bpms[-self.buffer_size//2:]
                    L = self.buffer_size
                    
            processed = np.array(self.data_buffer)
            
#以下算疲勞           
        if g_f!=0:
            self.times2.append(time.time() - self.t0)
            self.data_buffer2.append(g_f)
        
        ##加入時間
            
            if N > self.buffer_size2:
                    self.data_buffer2 = self.data_buffer2[-self.buffer_size2:]
                    self.times2 = self.times2[-self.buffer_size2:]
                    N = self.buffer_size2
                    
            processed2 = np.array(self.data_buffer2)
    #眨眼                     
        leftEAR = self.eye_ratio(leftEye)
        rightEAR = self.eye_ratio(rightEye)
        self.blinks_counter, self.blinks = self.theBlinks(leftEAR,rightEAR,self.blinks_counter,self.blinks)
        
#以上算疲勞     

            #if點足夠的話開始跑
        if L == self.buffer_size:

                self.fps = float(L) / (self.times[-1] - self.times[0])
                even_times = np.linspace(self.times[0], self.times[-1], L)
                                              
                #ICA
                #A_=self.ICA(processed)
                #processed =A_[0]                
                #基線飄移
                try:
                    processed = signal.detrend(processed)       
                except:
                    return
                interpolated = np.interp(even_times, self.times, processed)                                 
                interpolated = np.hamming(L) * interpolated               
                norm = (interpolated - np.mean(interpolated))/np.std(interpolated)
                                
                norm = interpolated/np.linalg.norm(interpolated)                
                #norm=self.smoothTriangle(norm)
                
    
                #帶通濾波器
                processed_samples=processed
                processed=self.butterworth_bandpass_filter(norm,0.8,3,self.fps,order=4)
                processed_samples=self.butterworth_bandpass_filter(processed_samples,0.8,3,self.fps,order=4)
            
                processed=signal.detrend(processed)
                processed_samples=signal.detrend(processed_samples)
                #歸一化
                avg = sum(processed)/len(processed)
                norm1=self.Z_ScoreNormalization(processed,avg,statistics.stdev(processed))
                
                #歸一化
                bpx=norm1
    
            
                for i in range(len(bpx)):
                    self.bp.append(bpx[i])
                    if len(self.now_time)<100:
                        self.now_time.append(self.time_buffer[i+1])
                    elif 100<=len(self.now_time)<200:
                        self.now_time.append(self.time_buffer[i+1]+self.now_time[99])
                    elif 200<=len(self.now_time)<300:
                        self.now_time.append(self.time_buffer[i+1]+self.now_time[199])
                    elif 300<=len(self.now_time)<400:
                        self.now_time.append(self.time_buffer[i+1]+self.now_time[299])
                
                self.bp=self.smoothTriangle(self.bp)
                
                #for i in range(len(self.bp)):
                    #s=str(self.bp[i])+'\n'
                    #print(s)
                    #file = open(r'C:\Users\a1016\.spyder-py3\python_FFT\BP1.txt', 'w+')
                    #file.write(s)   # 把資料寫入檔案
                    #file.close()   # 關閉檔案
                #for i in range(len(self.now_time)):
                    #s=str(self.fft)+'\n'
                    #print(s)
                    #file = open(r'C:\Users\a1016\.spyder-py3\python_FFT\TIME1.txt', 'w+')
                    #file.write(s)   # 把資料寫入檔案
                    #file.close()
                
                if self.now_time[0]>=300:
                    
                    bpo,now_timeo=self.BP.run(self.bp,self.now_time)
                    self.bpolist.append(bpo)
                    self.now_timeoliset.append(now_timeo)
                    #print(self.bpolist)
                    if len(self.bpolist)>15:
                        self.SBP_DBP,self.SBP_DBP2=statistics.mean(self.bpolist),statistics.mean(self.now_timeoliset)
                        self.bpolist.clear()
                        self.now_timeoliset.clear()
            
                if len(self.bp)>300:
                       for i in range(len(processed)):
                           self.bp.pop(0)
                       for X_time in range(len(processed)):
                           self.now_time.pop(0)
   
                #一階差分
                diff1=[self.bp[i]-self.bp[i+1] for i in range(len(self.bp)-1)]
                for x in range(len(diff1)):
                    self.x_derivative1.append(diff1[x])
                    
                if len(self.x_derivative1)>299:
                    self.now_time2.clear()
                    #for i in range(len(self.bp)): 
                    for i in range(299):
                           self.x_derivative1.pop(0)
                           
                if len(self.x_derivative1)==299:
                    self.now_time2=[]
                    for i in range(len(self.now_time)-1):
                        self.now_time2.append(self.now_time[i+1])
                if self.SBP_DBP > 0 and self.SBP_DBP2 > 0 :        
                    self.SBP_DBP_data.append(self.SBP_DBP)
                    self.SBP_DBP2_data.append(self.SBP_DBP2)        
                        
                        
                        
                #從這裡開始改        
                        
                raw = np.fft.rfft(norm*100)                
                self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)                
                freqs = 60. * self.freqs                               
                self.fft = np.abs(raw)**2           
                idx = np.where((freqs > 60) & (freqs < 100))             
                #sidx2=str(idx2)+'\n'
                #print(sidx2)
                               
                pruned = self.fft[idx]               
                pfreq = freqs[idx]
                
                 
                self.freqs = pfreq 
                self.fft = pruned


                #以下測試用
                #sfreqs2=str(self.freqs2)+'\n'
                #print("freqs2")
                #print(sfreqs2)
                #yyy=self.freqs2[0]
                #print(yyy)
                #file = open(r'C:\Users\a1016\.spyder-py3\python_FFT\freqs2.txt', 'w+')
                #file.write(sfreqs2)
                #file.close()
                
                #sfft2=str(self.fft2)+'\n'
                #print("fft2")
                #print(sfft2)
                #www=self.fft2[0]
                #print(www)
                #file = open(r'C:\Users\a1016\.spyder-py3\python_FFT\fft2.txt', 'w+')
                #file.write(sfft2)
                #file.close()
                #以上測試用
                
                             
                try:
                    idx_hr=np.argmax(pruned)
                
                    self.bpm = self.freqs[idx_hr]
                except:
                    self.bpm = self.bpm
                #self.bpm = math.trunc(round(self.bpm,0))
                
                if len(self.bpms)<100:
                    self.bpms.append(self.bpm)
                    
                '''if (len(self.bpms)>50 and len(self.mode_array)<100):   
                    if (max(self.bpms-np.mean(self.bpms)<5)): 
                        bpms=math.trunc(round(np.mean(self.bpms)))                                                        
                        self.mode_array.append(bpms)
                    
                if len(self.mode_array)==100:
                    vals,counts = np.unique(self.mode_array, return_counts=True)
                    mode=np.argmax(counts)                    
                    self.hr=vals[mode]'''
                    
                    
                self.samples = processed_samples
                
#以下算疲勞
        
        
        
        if N == self.buffer_size2:
                
                self.fps2 = float(N) / (self.times2[-1] - self.times2[0])
                even_times2 = np.linspace(self.times2[0], self.times2[-1], N)
                #ICA
                #A2_=self.ICA(processed2)
                #processed2 =A2_[0] 
                #基線飄移
                try:
                    processed2 = signal.detrend(processed2)
                except:
                    return
                interpolated2 = np.interp(even_times2, self.times2, processed2)
                #interpolated2 = np.hamming(N) * interpolated2
                #norm2 = (interpolated2 - np.mean(interpolated2))/np.std(interpolated2)
                norm2 = interpolated2/np.linalg.norm(interpolated2)
                #norm2=self.smoothTriangle(norm2)
                
                raw2 = np.fft.rfft(norm2)
                self.freqs2 = float(self.fps2) / N * np.arange(N / 2 + 1)
                self.fft2 = np.abs(raw2)**2
                		
                
        #LFHF
                fatigue_interval=1 #隔n-1幾次再進行運算
                if(self.count%fatigue_interval==0):
                    self.fatigue_time1 = self.fatigue_time2
                    self.fatigue_time2 = time.time()
                    if self.count/fatigue_interval==0:
                        self.total_time=time.time()-self.start_time
                    dtime = self.fatigue_time2 - self.fatigue_time1
                    self.time_array.append(dtime)
                    #sfatigue_time1=str(self.fatigue_time1)+'\n'
                    #print("fatigue_time1")
                    #print(sfatigue_time1)
                    #sfatigue_time2=str(self.fatigue_time2)+'\n'
                    #print("fatigue_time2")
                    #print(sfatigue_time2)
                    
                    
                    self.lf, self.hf = self.theArea(self.freqs2,self.fft2)
                
                    
                    self.lfhf=self.lf/self.hf
                    
                    self.dlfhf=(self.lfhf-self.last_hflf)/dtime
                    self.last_hflf=self.lfhf
                     
                    #if self.lfhf<=1:                        
                        #return
                    
                    self.lf_data.append(self.lf)
                    self.hf_data.append(self.hf)                     
                    self.lfhf_data.append(self.lfhf)    
                    
                    self.lf_array = self.lf_data
                    self.hf_array = self.hf_data
                    self.lfhf_array = self.lfhf_data
                    '''
                    self.lf_array.append(np.mean(self.lf_data))
                    self.hf_array.append(np.mean(self.hf_data))
                    self.lfhf_array.append(np.mean(self.lfhf_data))
                    '''
                    self.dlfhf_data.append(self.dlfhf)
                    
                    #self.SBP_DBP_data.append(self.SBP_DBP)
                    #self.SBP_DBP2_data.append(self.SBP_DBP2)
                    
                    self.count=self.count+1
                    
                else:
                    self.count=self.count+1
                    
        #熵
                if self.count%50==0 and len(self.BP.SampEn_timedata)>500:
                    
                    
                    SampEn_L=self.BP.SampEn_timedata[-500:]
                    self.SampEn=self.SampEn_calculation(SampEn_L,1,0.15)   
                    self.SampEn_data.append(self.SampEn)
                    
                    ApEn_U=SampEn_L
                    self.ApEn_U=self.ApEn_calculation(ApEn_U,1,0.15)   
                    self.ApEn_data.append(self.ApEn_U)
                
  
       
                        
                        
        #寫入數據資料   
                if (self.count/fatigue_interval)%int(self.subject_count)==0: #蒐集n筆資料 
                
                    
                    
                    filename = self.theFilename(self.subject_filename)
                    
                    self.total_time_file=time.time()-self.start_time    
                    self.now_fatigue_time=time.ctime() 
                    height = int(self.subject_height)/100
                    weight = int(self.subject_weight)
                    BMI = float(weight/(height*height))
                    
                    result_data=[self.subject_num,self.subject_gender,int(self.subject_age),
                                 float(self.subject_height),float(self.subject_weight),BMI,
                                 float(self.subject_sitting_height),float(self.subject_luminance),
                                 "","","",
                                 self.GUI_bpms,self.SBP_DBP,self.SBP_DBP2,self.blinks,
                                 np.mean(self.lf_array),np.mean(self.hf_array),np.mean(self.lfhf_array),
                                 np.mean(self.lf_area_data),np.mean(self.hf_area_data),
                                 self.total_time,self.total_time_file,self.now_fatigue_time]
                    
                    
                    self.file = saving.saving(filename,result_data,self.subject_num,self.lf_data,self.hf_data,self.lfhf_data,
                                              self.lf_array,self.hf_array,self.lfhf_array,
                                              self.dlfhf_data,self.SampEn_data,self.ApEn_data,
                                              self.SBP_DBP_data,self.SBP_DBP2_data)
                    
                    self.subject_luminance=100
                    index = [np.mean(self.lfhf_array),np.mean(self.dlfhf_array),np.mean(self.SampEn_data),np.mean(self.ApEn_data),
                             float(self.subject_luminance),self.GUI_bpms,np.mean(self.SBP_DBP_data),np.mean(self.SBP_DBP2_data)]
                    self.ann_predict = self.the_ANN(index)
                    self.file = False 
                    
                    
                    
                    self.lf_data = []
                    self.hf_data = []
                    self.lf_area_data = []
                    self.hf_area_data = []
                    self.lfhf_data = []
                    self.dlfhf_data = []
                    self.SampEn_data = []
                    self.ApEn_data = []
                    self.SBP_DBP_data = []
                    self.SBP_DBP2_data = [] 
        
#以上算疲勞
        if int(self.subject_count)>0:
            the_count = self.count % int(self.subject_count)                 
            self.progress = 100*(N/5+the_count)/(N/5+int(self.subject_count))
        elif int(self.subject_count)==0 and N>0:
            self.progress = (N/self.buffer_size2)*100
        
        if(mask.shape[0]!=10): 
            out = np.zeros_like(face_frame)
            mask = mask.astype(np.bool)
            out[mask] = face_frame[mask]
            if(processed[-1]>np.mean(processed)):
                out[mask,2] = 180 + processed[-1]*10
            face_frame[mask] = out[mask]
                        
    def reset(self):
        self.frame_in = np.zeros((10, 10, 3), np.uint8)
        self.frame_ROI = np.zeros((10, 10, 3), np.uint8)
        self.frame_out = np.zeros((10, 10, 3), np.uint8)
        self.samples = []
        self.times = [] 
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.fft2 = []
        self.freqs = []
        self.freqs2 = []
        self.t0 = time.time()
        self.bpm = 0
        self.bpms = []
        self.hf=0
        self.lf=0
        self.lfhf=0
        self.hfnu=[]
        self.lfnu=[]
        self.lfhfnu=[]
        self.dlfhf=0
    ##巴特濾波器
    def butterworth_bandpass(self, lowcut, highcut, fs, order=1):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band',analog='True')
        return b, a
        #order 是濾波器的階數，階數越大，濾波效果越好，但是計算量也會跟著變大。
        #所產生的濾波器參數 a 和 b 的長度，等於 order+1。
        #Wn 是正規化的截止頻率，介於 0 和 1 之間，當取樣頻率是 fs 時，所能處理的
        #最高頻率是 fs/2，所以如果實際的截止頻率是 f = 1000，那麼 Wn = f/(fs/2)。
        #function 是一個字串，function = 'low' 代表是低通濾波器，function = 'high' 代表是高通濾波。
        #fs=12,wn=f/(fs/2),如果截止頻率大於6,就高於正規化的截止頻率

    def butterworth_bandpass_filter(self, data, lowcut, highcut, fs, order=1):
        b, a = self.butterworth_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y
    
    def SampEn_calculation(self, L, m, r):
        N = len(L)
        B = 0.0
        A = 0.0
    
    
        # Split time series and save all templates of length m
        xmi = np.array([L[i : i + m] for i in range(N - m)])
        xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

        # Save all matches minus the self-match, compute B
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

        # Similar for computing A
        m += 1
        xm = np.array([L[i : i + m] for i in range(N - m + 1)])

        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

        # Return SampEn
        return -np.log(A / B)
    
    
    def ApEn_calculation(self,U, m, r):
        
        def maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        
        def phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [
                len([1 for x_j in x if maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
                for x_i in x
            ]
            return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
        N = len(U)
    
        return phi(m) - phi(m + 1)
    
    
            
    #計算LFHF面積
    
    def theArea(self,freqs2,fft2):       
       
        
        idx_LF = np.where((freqs2 >=0.04) & (freqs2 <= 0.15))                
        pruned_LF =fft2[idx_LF]
        pfreq_LF = freqs2[idx_LF]
        freqs_LF = pfreq_LF
        fft2_LF = pruned_LF
        
        idx_HF = np.where((freqs2 > 0.15) & (freqs2 < 0.4))                
        pruned_HF =fft2[idx_HF]
        pfreq_HF = freqs2[idx_HF]
        freqs_HF = pfreq_HF
        fft2_HF = pruned_HF
        
        
        
        lf_area=np.trapz(fft2_LF,freqs_LF)
        hf_area=np.trapz(fft2_HF,freqs_HF)
        
        lfnu=lf_area/(hf_area+lf_area)*100
        hfnu=hf_area/(hf_area+lf_area)*100
      
        return lfnu,hfnu
        
    
    def theBlinks(self,leftEAR,rightEAR,blinks_counter,blinks):
        
        eye_ar_thresh = 0.3 #眨眼閾值
        eye_ar_consec_frames = 3 #眨眼閾值
        
        
        if leftEAR!=0 and rightEAR!=0:       
            ear = (leftEAR + rightEAR) / 2.0 #平均左右眼縱橫比
            if ear < eye_ar_thresh:
                blinks_counter += 1
            else:   
                if blinks_counter >= eye_ar_consec_frames:
                    blinks += 1                                
                blinks_counter = 0
                
        return blinks_counter,blinks
    
    def eye_ratio(self,eye): #人眼縱橫比(eye aspect ratio)
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        if C == 0:
            return 0
        ear = (A + B) / (2.0 * C) 
        return ear



    def theFilename(self,subject_filename):
        
        loc_dt = datetime.datetime.today() 
        
        
        if subject_filename == 1:#依日期建檔
            loc_dt_format = loc_dt.strftime("%Y_%m%d")
            date = str(loc_dt_format)
            filename = date
            
        elif subject_filename == 2:#依月份寫檔
            loc_dt_format = loc_dt.strftime("%Y_%m")
            filename_month = str(loc_dt_format)
            filename = filename_month
            
        elif subject_filename == 3:#依年份寫檔
            loc_dt_format = loc_dt.strftime("%Y")
            filename_year = str(loc_dt_format)
            filename = filename_year
            
        elif subject_filename == 0:#檔名已經設定完成寫檔
            
            filename = "測量結果"
        
        return filename
    
    def the_ANN(self,index):
        sc = StandardScaler()
        ann = load_model('ann.h5')
        data = pd.read_csv("ALL.csv")
        X = data.iloc[:,0:8].values
        X = sc.fit_transform(X)
        
        ann_predict = np.argmax(ann.predict(sc.transform([index])))
        return ann_predict
            
        
        

    







































