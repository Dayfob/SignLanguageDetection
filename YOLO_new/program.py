import torch
import numpy as np
import cv2
from spellchecker import SpellChecker
from gtts import gTTS
import datetime
import os

model = torch.hub.load('ultralytics/yolov5',
                       'custom',
                       path='yolov5/runs/train/exp9/weights/best.pt',
                       force_reload=True)

cap = cv2.VideoCapture(0)
acc = ''
acc_size = 0
acc_control_size = 30
word = ""
sentence = []

while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if not results.pandas().xyxy[0].empty:
        letter = results.pandas().xyxy[0]["name"].iloc[0]
        if acc == '' or acc == letter:
            acc_size += 1

            if acc_size == acc_control_size:
                if not word or (word and word[-1] != letter):
                    if letter == '_':
                        if len(sentence) == 0 or sentence[-1] != word:
                            sentence.append(word)
                            print(word)
                        word = ""
                    else:
                        word += letter
                acc_size = 0
                acc = ''
        else:
            acc = letter

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


spell = SpellChecker()
output = ""

for word in sentence:
    if word == '':
        continue
    output += spell.correction(word) + ' '

output = output.lower()

print(output)

language = 'en'
recording_name = "recording.mp3"

recording = gTTS(text=output, lang=language, slow=False)

recording.save(recording_name)
os.system(recording_name)

