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

def disparityToPointCloud(disparity, Q):
    # Reproject disparity to 3D space
    points_3D = cv.reprojectImageTo3D(disparity, Q)
    return points_3D

# ----------------- Main ----------------- #
client = RemoteAPIClient()
sim = client.require('sim')
sim.startSimulation()


# Get vision sensor handle
visionSensor = sim.getObject('./visionSensor')
print(f'visionSensor handle : {visionSensor}')

disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0  # Create disparity map
Q = ...  # This needs to be loaded or calculated from stereo calibrationb:  https://docs.opencv.org/4.11.0/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6 
points_3D = disparityToPointCloud(disparity, Q)

while True:
    img = readFrame()
    img = filterTable(img)
    cv.imshow('img', img)
    cv.waitKey(1)