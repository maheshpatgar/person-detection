from ultralytics import YOLO
import cv2, os, requests
from datetime import datetime

FIREBASE_URL ="https://snapshots-db7da-default-rtdb.asia-southeast1.firebasedatabase.app/"

model, cap= YOLO('yolov8n.pt'), cv2.VideoCapture(0)
os.makedirs('snapshots', exist_ok=True)
last_save=0
last_firebase_update=0

def send_to_firebase(person_detected):
    """send person detection status to firebase realtime database"""
    try:
        data = {
            "person_detected":1 if person_detected else 0,"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        }
        response = requests.put (f"{FIREBASE_URL}/detection.json",json=data)
        if response.status_code==200:
            print(f"Firebase updated: person_detected = {1 if person_detected else 0}")
        else:
            print(f"firebase error: {response.status_code}") 
    except Exception as e:
        print(f"firebase exception: {e}")



while True:
    ret, frame= cap.read()
    if not ret: break

    r = model(frame)[0]
    person_count= sum(int(b.cls[0])==0 for b in r.boxes)
    frame = r.plot()
    cv2.putText(frame, f"Persons: {person_count}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)

    if (person_count>0 and (datetime.now().timestamp() - last_firebase_update) >= 5):
        send_to_firebase(person_detected=True)
        last_firebase_update= datetime.now().timestamp()

    if person_count>0 and (datetime.now().timestamp() - last_save) >= 3:
        timestamp= datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.rectangle(frame,(0,0),(460,60),(0,0,0),-1)
        cv2.putText(frame,f"{timestamp}| persons: {person_count}",(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imwrite(f"snapshots/{datetime.now().strftime('%Y%M%D_%H%M%S')}.jpg", frame)
        print(f"saved snapshot")
        last_save= datetime.now().timestamp()

    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    send_to_firebase(False)
cap.release()
cv2.destroyAllWindows()