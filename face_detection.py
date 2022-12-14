import cv2
from cv2 import cuda
import numpy as np
import dlib
from imutils import face_utils
import imutils
import os




class FaceDetection(object):
    def __init__(self):
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=200)
        self.fps2=0
        
    
    def face_detect(self, frame):
        
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        leftEye = np.zeros((6, 2), np.uint8)
        rightEye = np.zeros((6, 2), np.uint8)
        status = False
        
        
        
        if frame is None:
            return 
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        gray = gpu_gray.download()
        
        rects = self.detector(gray, 0)
        

        if len(rects)>0:
            status = True
            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
            (x, y, w, h) = face_utils.rect_to_bb(rects[0])
            
            if y<0:
                return frame, face_frame, ROI1, ROI2, status, mask, leftEye, rightEye
            face_frame = frame[y:y+h,x:x+w]
            
            if(face_frame.shape[:2][1] != 0):
                face_frame = imutils.resize(face_frame,width=256)
            
            face_frame = self.fa.align((frame),(gray),(rects[0])) 
            
            gpu_face_frame = cv2.cuda_GpuMat()
            gpu_face_frame.upload(face_frame)
            gpu_grayf = cv2.cuda.cvtColor(gpu_face_frame, cv2.COLOR_BGR2GRAY)
            grayf = gpu_grayf.download()
            rectsf = self.detector(grayf, 0)
            
            try:
                shape = self.predictor(grayf, rectsf[0])
                shape = face_utils.shape_to_np(shape)
            #????????????
                leftEye = shape[lStart:lEnd] #??????????????????
                rightEye = shape[rStart:rEnd]
            except:
                return frame, face_frame, ROI1, ROI2, status, mask, leftEye, rightEye
            if len(rectsf) >0:
                
                for (a, b) in shape:
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1) 
                    
                    #draw facial landmarks
                    #cv2.circle(??????, ????????????, ??????, ??????, ????????????)
                    #?????????????????????
                    
                #1????????????
                '''cv2.rectangle(face_frame,(shape[54][0], shape[29][1]), 
                      (shape[12][0],shape[33][1]), (0,255,0), 0)
                
                cv2.rectangle(face_frame, (shape[4][0], shape[29][1]), 
                       (shape[48][0],shape[33][1]), (0,255,0), 0)                
                ROI1 = face_frame[shape[29][1]:shape[33][1], #right cheek
                       shape[54][0]:shape[12][0]]
                        
                ROI2 =  face_frame[shape[29][1]:shape[33][1], #left cheek
                     shape[4][0]:shape[48][0]]   '''

                
                
                #draw rectangle on right and left cheeks
                #cv2.rectangle(??????, ????????????, ??????????????????, ??????, ????????????)
                #???????????????
                
                    
                
               
                
                #2??????
                cv2.rectangle(face_frame,(shape[64][1], shape[67][1]),
                   (shape[20][1],shape[6][1]), (0,255,0), 0)

                cv2.rectangle(face_frame, (shape[64][1], shape[67][1]), 
                   (shape[20][1],shape[6][1]), (0,255,0), 0)
                
                
                ROI1 = face_frame[shape[29][1]:shape[33][1], #right cheek
                    shape[54][0]:shape[12][0]]
                
                ROI2 = face_frame[shape[29][1]:shape[33][1], #left cheek
                      shape[4][0]:shape[48][0]]    
                
                #3????????????????????????
                '''cv2.rectangle(face_frame,(shape[0][0], shape[28][1]), 
                       (shape[26][0],shape[9][1]), (0,255,0), 0)
            
                cv2.rectangle(face_frame, (shape[0][0], shape[28][1]), 
                       (shape[26][0],shape[9][1]), (0,255,0), 0)
                ROI1 = face_frame[shape[28][1]:shape[9][1], 
                     shape[0][0]:shape[26][0]]
                ROI2 =  face_frame[shape[28][1]:shape[9][1], 
                   shape[0][0]:shape[26][0]]   '''

                #0???x???
                #1???y???
                #get the shape of face for color amplification
                rshape = np.zeros_like(shape) 
                rshape = self.face_remap(shape)    
                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))
            
                cv2.fillConvexPoly(mask, rshape[0:27], 1) 
                #???????????????void fillConvexPoly(Mat& img, const Point* pts, int npts, const Scalar& color, int lineType= 8 , int shift= 0 )
                #?????????????????????????????????
                #???????????????img ??????
                #pts ????????????????????????????????????
                #npts ????????????????????????
                #color ??????????????????
                #LineType ?????????????????????????????????
                                  #8 (or 0 ) - 8 - connected line???8??????)????????????
                                  #4 - 4 - connected line(4??????)????????????
                                  #CV_AA - antialiased?????????
                #shift??????????????????????????????
                #?????????????????????fillConvexPoly???????????????????????????
                #?????????????????????cvFillPoly?????????
                #?????????????????????????????????????????????????????????????????????????????????
                #????????????????????????????????????????????????????????????????????????
                # mask = np.zeros((face_frame.shape[0], face_frame.shape[1],3),np.uint8)
                # cv2.fillConvexPoly(mask, shape, 1)


                
        else:
            cv2.putText(frame, "No face detected",
                       (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            status = False
        return frame, face_frame, ROI1, ROI2, status, mask, leftEye, rightEye
    
    # some points in the facial landmarks need to be re-ordered
    
    def face_remap(self,shape):
        remapped_image = shape.copy()
        # left eye brow
        remapped_image[17] = shape[26]
        remapped_image[18] = shape[25]
        remapped_image[19] = shape[24]
        remapped_image[20] = shape[23]
        remapped_image[21] = shape[22]
        # right eye brow
        remapped_image[22] = shape[21]
        remapped_image[23] = shape[20]
        remapped_image[24] = shape[19]
        remapped_image[25] = shape[18]
        remapped_image[26] = shape[17]
        # neatening 
        remapped_image[27] = shape[0]
        
        remapped_image = cv2.convexHull(shape)
        return remapped_image
    

    
    
