from coppeliasim_zmqremoteapi_client import *
import numpy as np
import cv2 as cv

# ----------------- Read vision sensor ----------------- #
def readFrame():
    #sim.handleVisionSensor(visionSensor)
    buf, res = sim.getVisionSensorImg(visionSensor)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(*res, 3)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

# ----------------- Filter out background ---------------- #
def filterTable(img):
    lower = np.array([72, 72, 72])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(img, lower, upper)
    result = cv.bitwise_and(img,img, mask=mask)
    return result


# ----------------- Main ----------------- #
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()

# Get vision sensor handle
visionSensor = sim.getObject('./visionSensor')
print(f'visionSensor handle : {visionSensor}')

while True:
    img = readFrame()
    img = filterTable(img)    
    cv.imshow('img', img)
    cv.waitKey(1)
    