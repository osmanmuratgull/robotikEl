import cv2
from cvzone.SerialModule import SerialObject
from cvzone.HandTrackingModule import HandDetector
import cvzone.HandTrackingModule
import mediapipe as mp
import mainGestures as cnt

      
class handDetector():
    def __init__(self,mode=False,maxHands=2,modelComplexity=1,detectionCon=0.5,trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.modelComplex,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img,draw=True):
        img = cv2.flip(img,1)
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:   
                '''
                
                    '''
                if draw:
                    self.mpDraw.draw_landmarks(img,handlms,self.mpHands.HAND_CONNECTIONS)

        return img
    
    def findPosition(self,img,handNo=0,draw=True):
        
        lmlist = []
        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]

            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                lmlist.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        return lmlist


cap=cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector=HandDetector(maxHands=1,detectionCon=0.7)
mySerial= SerialObject("COM3",9600,1)

if not cap.isOpened():
    print('Could not open the camera')
else:
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    while True:
        
        
        ret,frame=cap.read()
        frame=cv2.flip(frame,1)
        hands,img=detector.findHands(frame)
        if hands:
            lmList=hands[0]
            fingerUp=detector.fingersUp(lmList)

            print(fingerUp)
            #cnt.led(fingerUp)
            if fingerUp==[0,0,0,0,0]:
                cv2.putText(frame,'Finger count:0',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            elif fingerUp==[0,1,0,0,0]:
                cv2.putText(frame,'Finger count:1',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)    
            elif fingerUp==[0,1,1,0,0]:
                cv2.putText(frame,'Finger count:2',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            elif fingerUp==[0,1,1,1,0]:
                cv2.putText(frame,'Finger count:3',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            elif fingerUp==[0,1,1,1,1]:
                cv2.putText(frame,'Finger count:4',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            elif fingerUp==[1,1,1,1,1]:
                cv2.putText(frame,'Finger count:5',(20,460),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA) 

        cv2.imshow("frame",frame)
        k=cv2.waitKey(1)
        if k==ord("k"):
            break

cap.release()
cv2.destroyAllWindows()
        
        
success,img=cap.read()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)        
img=detector.findHands(img)
lmList,bbox=detector.findDistance(img)
if lmList:
    fingers =detector.fingersUp()
    print(fingers)
mySerial.sendData(fingers)
cv2.imshow("Image",img)
cv2.waitKey(1)
        
  