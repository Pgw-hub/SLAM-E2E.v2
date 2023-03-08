import argparse
import sys
import serial
import time
import threading
import queue
import cv2
from PIL import Image
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as T

import imageio as iio
import matplotlib.pyplot as plt
import base64
from io import BytesIO

#import yolo

model = None
prev_image_array = None

transformations = T.Compose(
    [T.Lambda(lambda x: (x / 127.5) - 1.0)])

import torch.nn as nn
import torch.nn.functional as F


class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.
        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)
        the convolution layers are meant to handle feature engineering.
        the fully connected layer for predicting the steering angle.
        the elu activation function is for taking care of vanishing gradient problem.
        """
        super(NetworkNvidia, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=10),
            nn.Linear(in_features=10, out_features=1)
        )

    def forward(self, input):
        """Forward pass."""
        input = input.view(input.size(0), 3, 70, 320)
        output = self.conv_layers(input)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)
        return output
"""
def openSerial(port, baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=1, xonxoff=False, rtscts=False, dsrdtr=False):
    ser = serial.Serial()

    ser.port = port
    ser.baudrate = baudrate
    ser.bytesize = bytesize
    ser.parity = parity
    ser.stopbits = stopbits
    ser.timeout = timeout
    ser.xonxoff = xonxoff
    ser.rtscts = rtscts
    ser.dsrdtr = dsrdtr

    ser.open()
    return ser
"""

def writePort(ser, data):
    print("send : ", data)
    ser.write(data)


ser = serial.Serial(

    port='/dev/ttyACM0',

    baudrate=9600,

    parity=serial.PARITY_NONE,

    stopbits=serial.STOPBITS_ONE,

    bytesize=serial.EIGHTBITS,

        timeout=0)

print(ser.portstr) #연결된 포트 확인.

if ser.isOpen() == False :
    ser.open()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Auto Driving')
    parser.add_argument(
        '--model',
        type=str,
        default='../checkpoint/model-a-100_1.h5',
        help='Path to model h5 file. Model should be on the same path.'
    )
    
    args = parser.parse_args()

    model = NetworkNvidia()

    try:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    except KeyError:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model = checkpoint['model']

    except RuntimeError:
        print("==> Please check using the same model as the checkpoint")
        import sys
        sys.exit()
    
    # 포트 설정
    PORT = '/dev/ttyACM0'
    # 연결

    cap = cv2.VideoCapture(2)
    #cap = cv2.VideoCapture("./dorm.mp4")

    if(cap.isOpened() == False):
        print("Unable to read camera feed")


    cur_angle = 0
    # cur_velocity = 2

    #ser.write(b's') #initialize
    #ser.flush()

    # ser.write(b'w')
    #ser.flush()

    ser.write(b'w')
    #ser.flush()

    count = 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    (x,y), (w,h) = (0,140), (640, 200)

    while True:

        ret, frame = cap.read()

        #cv2.imshow("frame",frame)

        #frame_roi = frame[y:y+h, x:x+w]
        
   # i#mg = frame

        #frame = yolo.detect(frame)
        cv2.imshow("autodrive", frame)
        cv2.imwrite('frame.jpeg', frame)
        
        with open('./frame.jpeg', 'rb') as frame:
            frame_str = base64.b64encode(frame.read())
        
        #python detect.py --source ./data/images/dorm.mp4
        #if count%33==0:


        #label = detect.run(source = "frame.jpeg")
        #print("LABEL:", label)
        #if label==True:
        #   print("!!!!!!!!!!!!PERSON!!!!!!!!!!!!!!!!")
        #    ser.write(b's') #initialize
        #    ser.flush()
           #cv2.waitKey(2000)
        #    ser.write(b'w')
           #ser.flush()
        
        image = Image.open(BytesIO(base64.b64decode(frame_str)))
        image = image.resize((320,160))
            #image = frame
            #print(image.format)
            #print(image.mode)
            #print(image.size)
        image_array = np.array(image.copy())
        image_array = image_array[65:-25, :, :]

            # transform RGB to BGR for cv2
        image_array = image_array[:, :, ::-1]
        image_array = transformations(image_array)
        image_tensor = torch.Tensor(image_array)
        image_tensor = image_tensor.view(1, 3, 70, 320)
        image_tensor = Variable(image_tensor)

        steering_angle = model(image_tensor).view(-1).data.numpy()[0] #angle
        # steering_velocity = model(image_tensor).view(-1).data.numpy()[1] #velocity

        print("\nprediction-angle:", steering_angle)
        # print("prediction-vel:", steering_velocity)
        print()

        #steering_angle 값을 quantization을 해야함
        #steering_angle = steering_angle-0.253
        
        steering_angle = steering_angle * 20

        print("actual angle : ", steering_angle)

        print()
        # print("cur _ vel : ", cur_velocity)
        # print("pred _ velocity : ", steering_velocity)


        diff_angle = steering_angle - cur_angle
        #diff_angle = diff_angle / 10
        diff_angle = int(diff_angle)
        print("difference : ", diff_angle)

        cur_angle = steering_angle

        ser.write(b'd')
        ser.flush()
        #if count % 5 == 0 :
        if diff_angle == 0: 
            #ser.write(b'b')
            cv2.waitKey(33)
            time.sleep(0.2)
            continue
        elif diff_angle > 0: #angle이 오른쪽으로 꺽여야함
            for i in range(diff_angle) :
                    # D 신호를 보내야 함
                    #d = b'd'

                ser.write(b'd')
                ser.flush()
                print("d signal was sent")
                print()
                time.sleep(0.2)
                    #time.sleep(0.2)
        else : # angle이 왼쪽으로 꺽여야 함
            for i in range(-diff_angle) :
                    # a 신호를 보내야 함
                    #a = b'a'
                ser.write(b'a')
                ser.flush()
                print("a signal was sent")
                print()
                time.sleep(0.2)
                    #time.sleep(0.2)
            #time.sleep(0.5)
        #else :
        #    ser.write(b'b')


        # #velocity 
        # # 수정했음 !!!!!!!!!!!!!!!!!!!!!!!!!!!!        
        # if steering_velocity >= 1.5: #speed 2
        #     if cur_velocity == 1:
        #         ser.write(b'w')
        #         ser.flush()
        #         print("W signal was sent")
        #         time.sleep(0.2)
        #     elif cur_velocity == 0:
        #         ser.write(b'w')
        #         ser.write(b'w')
        #         ser.flush()
        #         print("W signal was sent")
        #         print("W signal was sent")
        #         time.sleep(0.2)
        #     cur_velocity = 2
        # elif steering_velocity >= 0.5: #speed 1
        #     if cur_velocity == 2:
        #         ser.write(b'x')
        #         ser.flush()
        #         print("X signal was sent")
        #         time.sleep(0.2)
        #     elif cur_velocity == 0:
        #         ser.write(b'w')
        #         ser.flush()
        #         print("W signal was sent")
        #         time.sleep(0.2)
        #     cur_velocity = 1
        # else: #speed 0
        #     #if cur_velocity == 1:
        #     ser.write(b's')
        #     ser.flush()
        #     print("S signal was sent")
        #     time.sleep(0.2)
        #     cur_velocity = 0


        count += 1
        cv2.waitKey(33)
    #cap.release()
    #cv2.destroyAllWindows()
    

