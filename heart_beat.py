import numpy as np
import cv2
import sys
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.PlotModule import LivePlot
import cvzone
import time

realWidth = 640   
realHeight = 480  
videoWidth = 160
videoHeight = 120
videoChannels = 3
videoFrameRate = 15

# Webcam Parameters
webcam = cv2.VideoCapture(1)
detector = FaceDetector()

webcam.set(3, realWidth)
webcam.set(4, realHeight)

# Color Magnification Parameters
levels = 4
alpha = 170
minFrequency = 1.0
maxFrequency = 2.0
bufferSize = 100
bufferIndex = 0



# Helper Methods
def buildGauss(frame, levels):
    pyramid = [frame]
    for level in range(levels):
        frame = cv2.pyrDown(frame)
        pyramid.append(frame)
    return pyramid

def reconstructFrame(pyramid, index, levels):
    filteredFrame = pyramid[index]
    for level in range(levels):
        filteredFrame = cv2.pyrUp(filteredFrame)
    filteredFrame = filteredFrame[:videoHeight, :videoWidth]
    return filteredFrame


# Output Display Parameters
font = cv2.FONT_HERSHEY_SIMPLEX
loadingTextLocation = (30, 40)
bpmTextLocation = (videoWidth//2, 40)
fpsTextLoaction = (500,600)

fontScale = 1
fontColor = (255,255,255)
lineType = 2
boxColor = (0, 255, 0)
boxWeight = 3

# Initialize Gaussian Pyramid
firstFrame = np.zeros((videoHeight, videoWidth, videoChannels))
firstGauss = buildGauss(firstFrame, levels+1)[levels]
videoGauss = np.zeros((bufferSize, firstGauss.shape[0], firstGauss.shape[1], videoChannels))
fourierTransformAvg = np.zeros((bufferSize))

# Bandpass Filter for Specified Frequencies
frequencies = (1.0*videoFrameRate) * np.arange(bufferSize) / (1.0*bufferSize)
mask = (frequencies >= minFrequency) & (frequencies <= maxFrequency)

# Heart Rate Calculation Variables
bpmCalculationFrequency = 10   #15
bpmBufferIndex = 0
bpmBufferSize = 10
bpmBuffer = np.zeros((bpmBufferSize))

i = 0
ptime = 0
ftime = 0
while (True):
    ret, frame = webcam.read()
    if ret == False:
        break

    frame, bboxs = detector.findFaces(frame,draw=False)
    frameDraw = frame.copy()
    ftime = time.time()
    fps = 1 / (ftime - ptime)
    ptime = ftime

    cv2.putText(frameDraw, f'FPS: {int(fps)}', (30, 440), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    if bboxs:
        x1, y1, w1, h1 = bboxs[0]['bbox']
        cv2.rectangle(frameDraw, bboxs[0]['bbox'], (255, 0, 255), 2)
        detectionFrame = frame[y1:y1 + h1, x1:x1 + w1]
        detectionFrame = cv2.resize(detectionFrame,(videoWidth,videoHeight))

        # Construct Gaussian Pyramid
        videoGauss[bufferIndex] = buildGauss(detectionFrame, levels+1)[levels]
        fourierTransform = np.fft.fft(videoGauss, axis=0)

        # Bandpass Filter
        fourierTransform[mask == False] = 0

        # Grab a Pulse
        if bufferIndex % bpmCalculationFrequency == 0:
            i = i + 1
            for buf in range(bufferSize):
                fourierTransformAvg[buf] = np.real(fourierTransform[buf]).mean()
            hz = frequencies[np.argmax(fourierTransformAvg)]
            bpm = 60.0 * hz
            bpmBuffer[bpmBufferIndex] = bpm
            bpmBufferIndex = (bpmBufferIndex + 1) % bpmBufferSize

        # Amplify
        filtered = np.real(np.fft.ifft(fourierTransform, axis=0))
        filtered = filtered * alpha

        # Reconstruct Resulting Frame
        filteredFrame = reconstructFrame(filtered, bufferIndex, levels)
        outputFrame = detectionFrame + filteredFrame
        outputFrame = cv2.convertScaleAbs(outputFrame)

        bufferIndex = (bufferIndex + 1) % bufferSize
        outputFrame_show = cv2.resize(outputFrame,(videoWidth//2,videoHeight//2))
        frameDraw[0:videoHeight // 2, (realWidth-videoWidth//2):realWidth] = outputFrame_show

        bpm_value = bpmBuffer.mean()

        if i > bpmBufferSize:
            cvzone.putTextRect(frameDraw,f'BPM: {bpm_value}',bpmTextLocation,scale=2)
        else:
            cvzone.putTextRect(frameDraw, "Calculating BPM...", loadingTextLocation,scale=2)

        if len(sys.argv) != 2:
            cv2.imshow("Heart Rate Monitor", frameDraw)
            cv2.imshow("output frame", outputFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        cv2.imshow("Heart Rate Monitor", frameDraw)
        cv2.imshow("output frame", outputFrame)
webcam.release()
cv2.destroyAllWindows()

