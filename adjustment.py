


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


hf = pd.read_csv("D:/Desktop/shake/hf.csv")
lf = pd.read_csv("D:/Desktop/shake/lf.csv")

#print(hf.columns)
#print(lf.columns)
hf_data = np.array(hf)
lf_data = np.array(lf)
first=68
last=69
for i in range (first,last):
    y_hf = hf_data[:1000,i]
    y_lf = lf_data[:1000,i]
     
    index_lf = []
    for a in range (len(y_lf)-1):
            if (y_lf[a] < y_lf[a+1])and(y_lf[a] < y_lf[a-1])and(y_lf[a]>51):
                index_lf.append(y_lf[a])   
    index_lf = np.array(index_lf)    
    lf_min = index_lf.min()
    
    index_hf = []
    for a in range (len(y_hf)-1):
            if (y_hf[a] > y_hf[a+1])and(y_hf[a] > y_hf[a-1])and(y_hf[a]<49):
                index_hf.append(y_hf[a])    
    index_hf = np.array(index_hf)    
    hf_max = index_hf.max()
    
    ###
    the_index_lf = np.full((len(y_lf)),lf_min)
    the_index_hf = np.full((len(y_hf)),index_hf.max())    
    x=np.arange(len(y_hf))
    plt.figure(figsize=(20,6))
    plt.plot(x, y_hf, color='r',linewidth=3)
    plt.plot(x, y_lf, color='b',linewidth=3)
    plt.plot(x, the_index_lf, color='black',linewidth=3)
    plt.plot(x, the_index_hf, color='black',linewidth=3)
    plt.xlabel("Sampling Point")
    plt.ylabel("Values")
    plt.title(i)
    plt.show()
    ###
    delete_lf = []
    for a in range (len(y_lf)):
        if y_lf[a]<lf_min:
           delete_lf.append(a)
           
    y_lf = np.delete(y_lf,delete_lf)
    
    delete_hf = []      
    for a in range (len(y_hf)):
        if y_hf[a]>hf_max:
            delete_hf.append(a)
    
    y_hf = np.delete(y_hf,delete_hf)
    
    ###
    x=np.arange(len(y_hf))
    plt.figure(figsize=(20,6))
    plt.plot(x, y_hf, color='r',linewidth=3)
    plt.plot(x, y_lf, color='b',linewidth=3)
    plt.xlabel("Sampling Point")
    plt.ylabel("Values")
    plt.title(i)
    plt.show()
    ###