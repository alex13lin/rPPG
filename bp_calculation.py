import numpy as np
import matplotlib.pyplot as plt
import heapq
import pandas as pd
import statistics
from scipy import signal

class BP(object):
    def __init__(self):
        self.SBP_DBP=[]
        self.SampEn_timedata=[]
        self.SampEn_time=0
    def closest(self,mylist, Number):
        answer = []
        for i in mylist:import numpy as np
        import matplotlib.pyplot as plt
        import heapq
        import pandas as pd
        import statistics
        from scipy import signal

        class BP(object):
            def __init__(self):
                self.SBP_DBP=[]
                self.SampEn_timedata=[]
                self.SampEn_time=0
            def closest(self,mylist, Number):
                answer = []
                for i in mylist:
                    answer.append(abs(Number-i))
                return answer.index(min(answer))
            
            #歸一化
            def Z_ScoreNormalization(self,x,mu,sigma):
                    x = (x - mu) / sigma;
                    return x;
                
            #抓一階倒數
            def get_derivative1(self,x,maxiter=1):   #//默認使用牛頓法?代10次
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
                
            def wave_guess(arr,t3):
                wn = int(len(arr)/4) 
                #?有經驗數據，先?置成1/4。
                #計算最小的N?值，也就是認為是波谷
                wave_crest = heapq.nlargest(wn, enumerate(arr), key=lambda x: x[1])
                wave_crest_mean = pd.DataFrame(wave_crest).mean()
            
                #計算最大的5?值，也認為是波峰
                wave_base = heapq.nsmallest(wn, enumerate(arr), key=lambda x: x[1])
                wave_base_mean = pd.DataFrame(wave_base).mean()
            
            
                print("######### result #########")
                #波峰，波谷的平均值的差，是波動週期。
                wave_period = abs(int( wave_crest_mean[0] - wave_base_mean[0]))
                print("wave_period_day:", wave_period)
                print("wave_crest_mean:", round(wave_crest_mean[1],2))
                print("wave_base_mean:", round(wave_base_mean[1],2))
            
            
            
                ############### 以下為畫圖顯示用 ###############
                wave_crest_x = [] #波峰x
                wave_crest_y = [] #波峰y
                for i,j in wave_crest:
                    wave_crest_x.append(i)
                    wave_crest_y.append(j)
                    
                wave_base_x = [] #波谷x
                wave_base_y = [] #波谷y
                for i,j in wave_base:
                    wave_base_x.append(i)
                    wave_base_y.append(j)
            
                #將原始數據和波峰，波谷畫到一張圖上
                plt.figure(figsize=(12,3))
                plt.plot(arr)
                #plt.plot(wave_base_x, wave_base_y, 'go')#藍色的點
                plt.plot(wave_crest_x, wave_crest_y, 'ro')#S紅色的點
                plt.grid()
                plt.show()
            def smoothTriangle(data, degree=1):
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
            def run(self,bp,t1):
                    #bp= np.arange(100)
                    #bp= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\BP.txt', delimiter='\n',max_rows =400)
                    #bp= np.around(x10_7,0)
                    bp=bp
                    
                    #t1= np.arange(100)
                    #t1= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\time1.txt', delimiter='\n',max_rows =400)
                    t1=t1
                    #d1= np.arange(99)
                    #d1= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\x_derivative1.txt', delimiter='\n',max_rows =297)
                    #bp= np.around(x10_7,0)
                    
                    #d2= np.arange(98)
                    #d2= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\x_derivative2.txt', delimiter='\n',max_rows =294)
                    #bp= np.around(x10_7,0)
                    #print(len(bp),' ',len(t1),'  ',len(d1),'  ',len(d2))
                    
                    #歸一化
                    avg = sum(bp)/len(bp)
                    bp =self.Z_ScoreNormalization(bp,avg,statistics.stdev(bp))
                    #時間
                    for i in t1:
                        i=i/10
                    #print(t1)
                    #t1=t1/10
                    bp=signal.detrend(bp)
                    #print('t1長度:',len(t1))
                    #一階差分刪掉第一個時間
                    t2=t1
                    t2=np.delete(t2,0)
                    #print('t2長度:',len(t2))
                    #二階差分刪掉1,2值
                    t3=t1
                    t3=np.delete(t3,0)
                    t3=np.delete(t3,0)
                    #print('t3長度:',len(t3))
                    
                    #一階導數
                    d1=self.get_derivative1(bp)
                    #一階差分
                    #d1=[bp[i]-bp[i+1] for i in range(len(bp)-1)]
                    #print('d1長度:',len(d1))
                    #二階差分
                    d2=[bp[i]-bp[i+1]-bp[i+2] for i in range(len(bp)-2)]
                    #print('d2長度:',len(d2))
                    
                    #二階差分抓最高值
                    num_peak_3 = signal.find_peaks(d2, distance=None)#distance表極大值點的距離至少大於等於10個水平單位
                    #print(len(num_peak_3))
                    
                    #print(len(num_peak_3[0]))
                    if len(num_peak_3[0])>=2:
                        #print(num_peak_3[0])
                        #print('the number of peaks is ' + str(len(num_peak_3[0])))
                        #輸入進陣列放到圖上
                        min1=[]
                        min1.append(num_peak_3[0][0])
                        for i in range(len(num_peak_3)):
                            if len(num_peak_3[0])==i:
                                min1.append(num_peak_3[0][i-1])
                                #print(min1)
                        
                            
                        #抓極值 二階差分最高值
                        #num_peak_3 = signal.find_peaks(d2, distance=None)
                        #print(num_peak_3[0])
                        #輸入進陣列放到圖上

                        max1_d2=[]
                        max1_t3=[]
                        
                        for i in range(len(num_peak_3[0])):
                          if len(num_peak_3[0])==i: 
                            max1_d2.append(d2[num_peak_3[0][0]])
                            max1_d2.append(d2[num_peak_3[0][0]])
                            max1_t3.append(t3[num_peak_3[0][i-1]])
                            max1_t3.append(t3[num_peak_3[0][i-1]])
                            #抓到一個週期的索引值 64~218
                            #print('A波位置',num_peak_3[0][2],'下一個',num_peak_3[0][3])
                            
                            
                            
                            #抓一階導數一個周期圖上的點
                            max1_d1=[]
                            max1_t1=[]
                            max1_d1.append(d1[num_peak_3[0][0]])
                            max1_d1.append(d1[num_peak_3[0][i-1]+1])
                            max1_t1.append(t1[num_peak_3[0][0]])
                            max1_t1.append(t1[num_peak_3[0][i-1]+1])
                        
                        '''
                        #抓一階差分一個周期
                        max1_d1=[]
                        max1_t2=[]
                        max1_d1.append(d1[num_peak_3[0][2]+1])
                        max1_d1.append(d1[num_peak_3[0][4]+1])
                        max1_t2.append(t2[num_peak_3[0][2]+1])
                        max1_t2.append(t2[num_peak_3[0][4]+1])
                        '''
                        
                        #抓一階倒數的波
                        
                        if len(num_peak_3[0])==2: 
                            Tbp=[d1[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][1]))]
                            #Tbp=smoothTriangle(Tbp)
                            TT1=[t2[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][1]))]
                            #print('Tbp長度:',len(Tbp))
                            #print('TT1長度:',len(TT1))
                        else:
                            for j in range(len(num_peak_3[0])):
                              if len(num_peak_3)==j: 
                                Tbp=[d1[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][j]))]
                                #Tbp=smoothTriangle(Tbp)
                                TT1=[t2[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][j]))]
                                #print('Tbp長度:',len(Tbp))
                                #print('TT1長度:',len(TT1))
                        
                        
                        #from scipy.interpolate import interp1d #注意是數字的1
                        #f1= interp1d(TT1,Tbp)               #產生線性插值函數
                        #print('max',max(TT1))
                        #print('min',min(TT1))
                        #x = np.linspace(1.7,2.67,100)             #將間隔細分為50個區段
                        #y = f1(x)                              #利用線性插值函數產生50個插值
                        #print(y)
                        #plt.plot(TT1,Tbp,'b^',x, y, "ro", label='linear interplot')
                        
                        
                        #64~218共155個值作為一個週期
                        #print('一階差分',len(TT1),len(Tbp))
                        

                        #抓波峰
                        num_peak_4 = signal.find_peaks(Tbp, distance=None)
                        max1_Tbp=[]
                        max1_TT1=[]
                        if len(num_peak_4[0])==1:
                            x=0
                            y=0
                            return x,y
                        max1_Tbp.append(Tbp[num_peak_4[0][0]])
                        max1_Tbp.append(Tbp[num_peak_4[0][1]])
                        max1_TT1.append(TT1[num_peak_4[0][0]])
                        max1_TT1.append(TT1[num_peak_4[0][1]])
                        #print('max',max1_Tbp)
                        #print('max時間',max1_TT1)
                        #print('波峰位置',num_peak_4[0][0],'下一個',num_peak_4[0][1])
                        
                        #for i in range(len(TT1)):
                        #    if TT1[i]==1.83274:
                        #        print(TT1[i],'第',i)
                        #    if TT1[i]>2.05 and TT1[i]<2.055:
                        #        print(TT1[i],'第',i)
                        #    if TT1[i]>2.3 and TT1[i]<2.31:
                        #        print(TT1[i],'第',i)
                        #    if TT1[i]>2.4 and TT1[i]<2.41:
                        #        print(TT1[i],'第',i)
                            
                        #print('Tbp長度:',len(Tbp))
                        num_zero=[]        
                        num_zero.append(self.closest(Tbp[0:35],0))
                        #num_zero.append(30+closest(Tbp[31:60],0))
                        #num_zero.append(61+closest(Tbp[61:120],0))
                        #for i in range(23,56):
                        #    num_zero=closest(Tbp[0:23],0)
                        #for i in range(56,96):
                        #    num_zero=closest(Tbp,0)
                        #for i in range(96,155):
                        #    num_zero=closest(Tbp,0)
                        #抓最接近0的值
                        #print(num_zero)
                        
                        zero_Tbp=[]
                        zero_TT1=[]
                        zero_Tbp.append(Tbp[num_zero[0]])
                        #zero_Tbp.append(Tbp[num_zero[1]])
                        #zero_Tbp.append(Tbp[num_zero[2]])
                        #zero_Tbp.append(Tbp[num_zero[3]])
                        zero_TT1.append(TT1[num_zero[0]])
                        #zero_TT1.append(TT1[num_zero[1]])
                        #zero_TT1.append(TT1[num_zero[2]])
                        #zero_TT1.append(TT1[num_zero[3]])
                        #print('接近零的',zero_Tbp)
                        #print('接近零的時間',zero_TT1)
                        
                        #min
                        #b = (np.diff(np.sign(np.diff(Tbp[:]))) > 0).nonzero()[0]+1
                        #https://tcoil.info/find-peaks-and-valleys-in-dataset-with-python/
                        minTbp_num=min(Tbp[50:100])
                        for i in range(len(Tbp[50:100])):
                            if Tbp[i+50]==minTbp_num:
                                b=i+50
                                
                        min_Tbp=[]
                        min_TT1=[]   
                        #[min_Tbp,min_Tbp_num] = findpeaks(Tbp)
                        min_Tbp.append(Tbp[b])
                        min_TT1.append(TT1[b])
                        #print(min_Tbp,min_TT1)
                        
                        #抓特徵點
                        feature_d1=[]
                        feature_Time=[]
                        
                        #起點
                        feature_d1.append(Tbp[0])
                        #第一個過零點
                        feature_d1.append(Tbp[num_zero[0]])
                        #第一個波峰
                        feature_d1.append(Tbp[num_peak_4[0][0]])
                        #波谷
                        feature_d1.append(Tbp[b])
                        #第二個波峰
                        feature_d1.append(Tbp[num_peak_4[0][1]])
                        #終點
                        feature_d1.append(Tbp[-1])
                        
                        #起點時間
                        feature_Time.append(TT1[0])
                        #第一個過零點時間
                        feature_Time.append(TT1[num_zero[0]])
                        #第一個波峰時間
                        feature_Time.append(TT1[num_peak_4[0][0]])
                        #波谷時間
                        feature_Time.append(TT1[b])
                        #第二個波峰時間
                        feature_Time.append(TT1[num_peak_4[0][1]])
                        #終點時間
                        feature_Time.append(TT1[-1])
                        
                        time1=TT1[-1]-TT1[0]
                        #print('時間',time1)
                        time2=60/time1
                        #print('乘幾秒',time2)
                        #SBP = -141.3 * (t2 / t) + 0.68 * HR + 145.6
                        #DBP =  -93.2 * (t2 / t) + 0.15 * HR + 120.6
                        #t = time1 ; t2 = (TT1(num_peak_4[0][1]-TT1[0])) ; HR = time1 * time2
                        SBP=-141.3*((TT1[num_peak_4[0][1]]-TT1[0])/(time1))+0.68*(time1)*time2+145.6
                        DBP=-93.2*((TT1[num_peak_4[0][1]]-TT1[0])/(time1))+0.15*(time1)*time2+120.6
                        #self.SBP_DBP=[SBP,'/',DBp]
                        #SBP_M01 = 9.6 * (t1 / t) - 38.0 * (t2 / t) + 145
                        #DEP_M01 = 8.7 * (t1 / t) - 16.0 * (t2 / t) + 86.2
                        SBP_M01  = 9.6 * ((TT1[num_peak_4[0][0]]-TT1[0])/(time1)) - 38 * ((TT1[-1]-TT1[num_peak_4[0][0]])/(time1)) + 145
                        DEP_M01  = 8.7 * ((TT1[num_peak_4[0][0]]-TT1[0])/(time1)) - 16 * ((TT1[-1]-TT1[num_peak_4[0][0]])/(time1)) + 86.2
                        #SBP_M02 = 115 * (t1 / t) - 142 * (t2 / t) + 153.2
                        #DEP_M02 = 31.1 *(t1 / t) - 24.7 * (t2 / t) + 0.556 * HR
                        SBP_M02  = 115 * ((TT1[num_peak_4[0][0]]-TT1[0])/(time1)) - 142  * ((TT1[-1]-TT1[num_peak_4[0][0]])/(time1)) + 153.2
                        DEP_M02  = 31.1 *((TT1[num_peak_4[0][0]]-TT1[0])/(time1)) - 24.7 * ((TT1[-1]-TT1[num_peak_4[0][0]])/(time1)) + 0.556*(time1)*time2
                        
                        self.SampEn_timedata.append(TT1[num_peak_4[0][1]]-self.SampEn_time)
                        self.SampEn_time=TT1[num_peak_4[0][1]]
                        
                        return SBP,DBP
                        '''
                        print (TT1[0])
                        print (TT1[-1])
                        print (TT1[num_peak_4[0][1]])
                        print (SBP)
                        print (DBP)
                        '''
                        #沒有0的值
                        #for i in range(len(Tbp)):
                        #    if Tbp[i]==0:
                        #        num_zero.append(i)
                        #        print('第幾個為0',i)   
                        #zero_Tbp.append(Tbp[num_zero[0]])
                        #zero_Tbp.append(Tbp[num_zero[1]])
                        #zero_TT1.append(TT1[num_zero[0]])
                        #zero_TT1.append(TT1[num_zero[1]])
                        
                        
                        
                        ##畫0線 進矩陣
                        #tbp1=[0 for i in range(len(d1))]
                        #tbp0=[0 for i in range(len(Tbp))]
                        '''
                        #抓很多點
                        #arr=d2
                        #wave_guess(arr,t3)
                        #BP
                        plt.figure(figsize=(12, 3))
                        plt.subplot(131)
                        plt.title('ppg')
                        plt.plot(t1,bp)
                        
                        #一階差分
                        plt.subplot(132)
                        #plt.title('first order difference')
                        plt.title('First Derivative')
                        plt.plot(t1,d1,'b',max1_t1,max1_d1,'r*',t1,tbp1,'k')
                        #plt.plot(t2,d1,'b',max1_t2,max1_d1,'r*')
                        
                        #二階差分
                        plt.subplot(133)
                        plt.title('second order difference')
                        plt.plot(t3,d2,'b',max1_t3,max1_d2,'r*')
                        #plt.plot(t3,d2,'b')
                        plt.show
                        
                        #週期波
                        plt.figure(figsize=(12, 3))
                        plt.title('First Derivative')
                        #plt.plot(TT1,Tbp,'b',TT1,tbp0,'k',max1_TT1,max1_Tbp,'r*')
                        #plt.plot(TT1,Tbp,'b',TT1,tbp0,'k',max1_TT1,max1_Tbp,'r*',zero_TT1,zero_Tbp,'g^')
                        plt.plot(TT1,Tbp,'b',TT1,tbp0,'k',feature_Time,feature_d1,'r*')
                        #換顏色的網址https://www.itread01.com/content/1548484217.html
                        '''
                    else:
                        x=0.0
                        y=0.0
                        return x,y 
                    
                    
            answer.append(abs(Number-i))
        return answer.index(min(answer))
    
    #歸一化
    def Z_ScoreNormalization(self,x,mu,sigma):
            x = (x - mu) / sigma;
            return x;
        
    #抓一階倒數
    def get_derivative1(self,x,maxiter=1):   #//默認使用牛頓法?代10次
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
        
    def wave_guess(arr,t3):
        wn = int(len(arr)/4) 
        #?有經驗數據，先?置成1/4。
        #計算最小的N?值，也就是認為是波谷
        wave_crest = heapq.nlargest(wn, enumerate(arr), key=lambda x: x[1])
        wave_crest_mean = pd.DataFrame(wave_crest).mean()
    
        #計算最大的5?值，也認為是波峰
        wave_base = heapq.nsmallest(wn, enumerate(arr), key=lambda x: x[1])
        wave_base_mean = pd.DataFrame(wave_base).mean()
    
    
        print("######### result #########")
        #波峰，波谷的平均值的差，是波動週期。
        wave_period = abs(int( wave_crest_mean[0] - wave_base_mean[0]))
        print("wave_period_day:", wave_period)
        print("wave_crest_mean:", round(wave_crest_mean[1],2))
        print("wave_base_mean:", round(wave_base_mean[1],2))
    
    
    
        ############### 以下為畫圖顯示用 ###############
        wave_crest_x = [] #波峰x
        wave_crest_y = [] #波峰y
        for i,j in wave_crest:
            wave_crest_x.append(i)
            wave_crest_y.append(j)
            
        wave_base_x = [] #波谷x
        wave_base_y = [] #波谷y
        for i,j in wave_base:
            wave_base_x.append(i)
            wave_base_y.append(j)
    
        #將原始數據和波峰，波谷畫到一張圖上
        plt.figure(figsize=(12,3))
        plt.plot(arr)
        #plt.plot(wave_base_x, wave_base_y, 'go')#藍色的點
        plt.plot(wave_crest_x, wave_crest_y, 'ro')#S紅色的點
        plt.grid()
        plt.show()
    def smoothTriangle(data, degree=1):
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
    def run(self,bp,t1):
            #bp= np.arange(100)
            #bp= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\BP.txt', delimiter='\n',max_rows =400)
            #bp= np.around(x10_7,0)
            bp=bp
            
            #t1= np.arange(100)
            #t1= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\time1.txt', delimiter='\n',max_rows =400)
            t1=t1
            #d1= np.arange(99)
            #d1= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\x_derivative1.txt', delimiter='\n',max_rows =297)
            #bp= np.around(x10_7,0)
            
            #d2= np.arange(98)
            #d2= np.loadtxt(r'C:\Users\user\.spyder-py3\python_FFT\TEST_5_16\x_derivative2.txt', delimiter='\n',max_rows =294)
            #bp= np.around(x10_7,0)
            #print(len(bp),' ',len(t1),'  ',len(d1),'  ',len(d2))
            
            #歸一化
            avg = sum(bp)/len(bp)
            bp =self.Z_ScoreNormalization(bp,avg,statistics.stdev(bp))
            #時間
            for i in t1:
                i=i/10
            #print(t1)
            #t1=t1/10
            bp=signal.detrend(bp)
            #print('t1長度:',len(t1))
            #一階差分刪掉第一個時間
            t2=t1
            t2=np.delete(t2,0)
            #print('t2長度:',len(t2))
            #二階差分刪掉1,2值
            t3=t1
            t3=np.delete(t3,0)
            t3=np.delete(t3,0)
            #print('t3長度:',len(t3))
            
            #一階導數
            d1=self.get_derivative1(bp)
            #一階差分
            #d1=[bp[i]-bp[i+1] for i in range(len(bp)-1)]
            #print('d1長度:',len(d1))
            #二階差分
            d2=[bp[i]-bp[i+1]-bp[i+2] for i in range(len(bp)-2)]
            #print('d2長度:',len(d2))
            
            #二階差分抓最高值
            num_peak_3 = signal.find_peaks(d2, distance=None)#distance表極大值點的距離至少大於等於10個水平單位
            #print(len(num_peak_3))
            
            #print(len(num_peak_3[0]))
            if len(num_peak_3[0])>=2:
                #print(num_peak_3[0])
                #print('the number of peaks is ' + str(len(num_peak_3[0])))
                #輸入進陣列放到圖上
                min1=[]
                min1.append(num_peak_3[0][0])
                for i in range(len(num_peak_3)):
                    if len(num_peak_3[0])==i:
                        min1.append(num_peak_3[0][i-1])
                        #print(min1)
                
                    
                #抓極值 二階差分最高值
                #num_peak_3 = signal.find_peaks(d2, distance=None)
                #print(num_peak_3[0])
                #輸入進陣列放到圖上

                max1_d2=[]
                max1_t3=[]
                
                for i in range(len(num_peak_3[0])):
                  if len(num_peak_3[0])==i: 
                    max1_d2.append(d2[num_peak_3[0][0]])
                    max1_d2.append(d2[num_peak_3[0][0]])
                    max1_t3.append(t3[num_peak_3[0][i-1]])
                    max1_t3.append(t3[num_peak_3[0][i-1]])
                    #抓到一個週期的索引值 64~218
                    #print('A波位置',num_peak_3[0][2],'下一個',num_peak_3[0][3])
                    
                    
                    
                    #抓一階導數一個周期圖上的點
                    max1_d1=[]
                    max1_t1=[]
                    max1_d1.append(d1[num_peak_3[0][0]])
                    max1_d1.append(d1[num_peak_3[0][i-1]+1])
                    max1_t1.append(t1[num_peak_3[0][0]])
                    max1_t1.append(t1[num_peak_3[0][i-1]+1])
                
                '''
                #抓一階差分一個周期
                max1_d1=[]
                max1_t2=[]
                max1_d1.append(d1[num_peak_3[0][2]+1])
                max1_d1.append(d1[num_peak_3[0][4]+1])
                max1_t2.append(t2[num_peak_3[0][2]+1])
                max1_t2.append(t2[num_peak_3[0][4]+1])
                '''
                
                #抓一階倒數的波
                
                if len(num_peak_3[0])==2: 
                    Tbp=[d1[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][1]))]
                    #Tbp=smoothTriangle(Tbp)
                    TT1=[t2[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][1]))]
                    #print('Tbp長度:',len(Tbp))
                    #print('TT1長度:',len(TT1))
                else:
                    for j in range(len(num_peak_3[0])):
                      if len(num_peak_3)==j: 
                        Tbp=[d1[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][j]))]
                        #Tbp=smoothTriangle(Tbp)
                        TT1=[t2[i] for i in range(int(num_peak_3[0][0]),int(num_peak_3[0][j]))]
                        #print('Tbp長度:',len(Tbp))
                        #print('TT1長度:',len(TT1))
                
                
                #from scipy.interpolate import interp1d #注意是數字的1
                #f1= interp1d(TT1,Tbp)               #產生線性插值函數
                #print('max',max(TT1))
                #print('min',min(TT1))
                #x = np.linspace(1.7,2.67,100)             #將間隔細分為50個區段
                #y = f1(x)                              #利用線性插值函數產生50個插值
                #print(y)
                #plt.plot(TT1,Tbp,'b^',x, y, "ro", label='linear interplot')
                
                
                #64~218共155個值作為一個週期
                #print('一階差分',len(TT1),len(Tbp))
                

                #抓波峰
                num_peak_4 = signal.find_peaks(Tbp, distance=None)
                max1_Tbp=[]
                max1_TT1=[]
                if len(num_peak_4[0])==1:
                    x=0
                    y=0
                    return x,y
                max1_Tbp.append(Tbp[num_peak_4[0][0]])
                max1_Tbp.append(Tbp[num_peak_4[0][1]])
                max1_TT1.append(TT1[num_peak_4[0][0]])
                max1_TT1.append(TT1[num_peak_4[0][1]])
                #print('max',max1_Tbp)
                #print('max時間',max1_TT1)
                #print('波峰位置',num_peak_4[0][0],'下一個',num_peak_4[0][1])
                
                #for i in range(len(TT1)):
                #    if TT1[i]==1.83274:
                #        print(TT1[i],'第',i)
                #    if TT1[i]>2.05 and TT1[i]<2.055:
                #        print(TT1[i],'第',i)
                #    if TT1[i]>2.3 and TT1[i]<2.31:
                #        print(TT1[i],'第',i)
                #    if TT1[i]>2.4 and TT1[i]<2.41:
                #        print(TT1[i],'第',i)
                    
                #print('Tbp長度:',len(Tbp))
                num_zero=[]        
                num_zero.append(self.closest(Tbp[0:35],0))
                #num_zero.append(30+closest(Tbp[31:60],0))
                #num_zero.append(61+closest(Tbp[61:120],0))
                #for i in range(23,56):
                #    num_zero=closest(Tbp[0:23],0)
                #for i in range(56,96):
                #    num_zero=closest(Tbp,0)
                #for i in range(96,155):
                #    num_zero=closest(Tbp,0)
                #抓最接近0的值
                #print(num_zero)
                
                zero_Tbp=[]
                zero_TT1=[]
                zero_Tbp.append(Tbp[num_zero[0]])
                #zero_Tbp.append(Tbp[num_zero[1]])
                #zero_Tbp.append(Tbp[num_zero[2]])
                #zero_Tbp.append(Tbp[num_zero[3]])
                zero_TT1.append(TT1[num_zero[0]])
                #zero_TT1.append(TT1[num_zero[1]])
                #zero_TT1.append(TT1[num_zero[2]])
                #zero_TT1.append(TT1[num_zero[3]])
                #print('接近零的',zero_Tbp)
                #print('接近零的時間',zero_TT1)
                
                #min
                #b = (np.diff(np.sign(np.diff(Tbp[:]))) > 0).nonzero()[0]+1
                #https://tcoil.info/find-peaks-and-valleys-in-dataset-with-python/
                minTbp_num=min(Tbp[50:100])
                for i in range(len(Tbp[50:100])):
                    if Tbp[i+50]==minTbp_num:
                        b=i+50
                        
                min_Tbp=[]
                min_TT1=[]   
                #[min_Tbp,min_Tbp_num] = findpeaks(Tbp)
                min_Tbp.append(Tbp[b])
                min_TT1.append(TT1[b])
                #print(min_Tbp,min_TT1)
                
                #抓特徵點
                feature_d1=[]
                feature_Time=[]
                
                #起點
                feature_d1.append(Tbp[0])
                #第一個過零點
                feature_d1.append(Tbp[num_zero[0]])
                #第一個波峰
                feature_d1.append(Tbp[num_peak_4[0][0]])
                #波谷
                feature_d1.append(Tbp[b])
                #第二個波峰
                feature_d1.append(Tbp[num_peak_4[0][1]])
                #終點
                feature_d1.append(Tbp[-1])
                
                #起點時間
                feature_Time.append(TT1[0])
                #第一個過零點時間
                feature_Time.append(TT1[num_zero[0]])
                #第一個波峰時間
                feature_Time.append(TT1[num_peak_4[0][0]])
                #波谷時間
                feature_Time.append(TT1[b])
                #第二個波峰時間
                feature_Time.append(TT1[num_peak_4[0][1]])
                #終點時間
                feature_Time.append(TT1[-1])
                
                time1=TT1[-1]-TT1[0]
                #print('時間',time1)
                time2=60/time1
                #print('乘幾秒',time2)
                SBP=-141.3*((TT1[num_peak_4[0][1]]-TT1[0])/(time1))+0.68*(time1)*time2+145.6
                DBP=-93.2*((TT1[num_peak_4[0][1]]-TT1[0])/(time1))+0.15*(time1)*time2+120.6
                #self.SBP_DBP=[SBP,'/',DBp]
                
                self.SampEn_timedata.append(TT1[num_peak_4[0][1]]-self.SampEn_time)
                self.SampEn_time=TT1[num_peak_4[0][1]]
                return SBP,DBP
                '''
                print (TT1[0])
                print (TT1[-1])
                print (TT1[num_peak_4[0][1]])
                print (SBP)
                print (DBP)
                '''
                #沒有0的值
                #for i in range(len(Tbp)):
                #    if Tbp[i]==0:
                #        num_zero.append(i)
                #        print('第幾個為0',i)   
                #zero_Tbp.append(Tbp[num_zero[0]])
                #zero_Tbp.append(Tbp[num_zero[1]])
                #zero_TT1.append(TT1[num_zero[0]])
                #zero_TT1.append(TT1[num_zero[1]])
                
                
                
                ##畫0線 進矩陣
                #tbp1=[0 for i in range(len(d1))]
                #tbp0=[0 for i in range(len(Tbp))]
                '''
                #抓很多點
                #arr=d2
                #wave_guess(arr,t3)
                #BP
                plt.figure(figsize=(12, 3))
                plt.subplot(131)
                plt.title('ppg')
                plt.plot(t1,bp)
                
                #一階差分
                plt.subplot(132)
                #plt.title('first order difference')
                plt.title('First Derivative')
                plt.plot(t1,d1,'b',max1_t1,max1_d1,'r*',t1,tbp1,'k')
                #plt.plot(t2,d1,'b',max1_t2,max1_d1,'r*')
                
                #二階差分
                plt.subplot(133)
                plt.title('second order difference')
                plt.plot(t3,d2,'b',max1_t3,max1_d2,'r*')
                #plt.plot(t3,d2,'b')
                plt.show
                
                #週期波
                plt.figure(figsize=(12, 3))
                plt.title('First Derivative')
                #plt.plot(TT1,Tbp,'b',TT1,tbp0,'k',max1_TT1,max1_Tbp,'r*')
                #plt.plot(TT1,Tbp,'b',TT1,tbp0,'k',max1_TT1,max1_Tbp,'r*',zero_TT1,zero_Tbp,'g^')
                plt.plot(TT1,Tbp,'b',TT1,tbp0,'k',feature_Time,feature_d1,'r*')
                #換顏色的網址https://www.itread01.com/content/1548484217.html
                '''
            else:
                x=0.0
                y=0.0
                return x,y 
            
            