from coppeliasim_zmqremoteapi_client import *
import numpy as np
import cv2 as cv


# ----------------- Read vision sensor ----------------- #
def readFrame(visionObject):
    #sim.handleVisionSensor(visionSensor)
    buf, res = sim.getVisionSensorImg(visionObject)
    img = np.frombuffer(buf, dtype=np.uint8).reshape(*res, 3)
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img

# ----------------- Filter out background ---------------- #
def filterTable(img):
    lower = np.array([72, 72, 72])
    upper = np.array([255, 255, 255])
    mask = cv.inRange(img, lower, upper)
    result = cv.bitwise_and(img,img, mask=mask)
    return result

#def disparityToPointCloud(disparity, Q):
    # Reproject disparity to 3D space
 #   points_3D = cv.reprojectImageTo3D(disparity, Q)
  #  return points_3D

# ----------------- Main ----------------- #
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()


# Get vision sensor handle
visionSensorright = sim.getObject('./visionSensorright')
visionSensorleft = sim.getObject('./visionSensorleft')

print(f'visionSensor handle : {visionSensorleft}')
print(f'visionSensor handle : {visionSensorright}')

stereo = cv.StereoBM_create(numDisparities=64,blockSize=15)

#Q = ...  # This needs to be loaded or calculated from stereo calibrationb:  https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6 
#points_3D = disparityToPointCloud(disparity, Q)

while True:
    imgL = readFrame(visionSensorleft)
    imgR = readFrame(visionSensorright)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0  # Create disparity map
    cv.imshow('imgL', imgL)
    cv.imshow('imgR', imgR)
    cv.imshow('disparity', disparity)
    cv.waitKey(1)