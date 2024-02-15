import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

detector = HandDetector(detectionCon=0.8, maxHands=2)
colorR = (255,0,255)

cx, cy, w, h  = 100, 100, 200, 200

class MoveRect():
    def __init__(self,posCenter,size = [200,200]):
        self.posCenter = posCenter
        self.size = size


    def updatePosition(self,cursor):
        cx = self.posCenter[0]
        cy = self.posCenter[1]
        w, h = self.size

        #if the index finger in the rect
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor

rect_list = []
for i in range(5):
    rect_list.append(MoveRect([i*250+150,150]))

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, flipType=False)  # with draw

    cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # list of 21 landmarks point
        bbox1 = hand1["bbox"]  # bounding box info list x, y, w, h
        centerPoint1 = hand1["center"]  # center of the hand
        handType1 = hand1["type"]  # left or right
        fingers1 = detector.fingersUp(hand1)


        if lmList1:

            d, _ , _  = detector.findDistance(lmList1[8][:2], lmList1[12][:2], frame)
            if d<25:
                cursor = lmList1[8] #index finger tip landmark
                for rect in rect_list:
                    rect.updatePosition(cursor)

    #for draw
    for rect in rect_list:
        cx = rect.posCenter[0]
        cy = rect.posCenter[1]
        w, h = rect.size
        cv2.rectangle(frame, (cx-w//2, cy-h//2), (cx+w//2, cy+h//2), colorR, cv2.FILLED)



    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()