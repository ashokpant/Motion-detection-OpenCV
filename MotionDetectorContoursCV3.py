import time
from datetime import datetime

import cv2 as cv
import numpy as np


class MotionDetectorAdaptative():
    def onChange(self, val):  # callback when the user change the detection threshold
        self.threshold = val

    def __init__(self, threshold=25, doRecord=True, showWindows=True):
        self.writer = None
        self.font = None
        self.doRecord = doRecord  # Either or not record the moving object
        self.show = showWindows  # Either or not show the 2 windows
        self.frame = None

        self.capture = cv.VideoCapture(0)
        _, self.frame = self.capture.read()  # Take a frame to init recorder
        self.height, self.width = self.frame.shape[:2]
        if doRecord:
            self.initRecorder()

        self.gray_frame = np.zeros((self.height, self.width), dtype=np.uint8)
        self.average_frame = np.zeros((self.height, self.width, 3), np.float32)
        self.absdiff_frame = None
        self.previous_frame = None

        self.surface = self.width * self.height
        self.currentsurface = 0
        self.currentcontours = None
        self.threshold = threshold
        self.isRecording = False
        self.trigger_time = 0  # Hold timestamp of the last detection

        if showWindows:
            cv.namedWindow("Image")
            cv.createTrackbar("Detection treshold: ", "Image", self.threshold, 100, self.onChange)

    def initRecorder(self):  # Create the recorder
        codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')  # ('W', 'M', 'V', '2')
        self.writer = cv.VideoWriter(filename=datetime.now().strftime("%b-%d_%H_%M_%S") + ".avi", fourcc=codec, fps=5,
                                     frameSize=(self.width, self.height), isColor=True)
        # FPS set to 5 because it seems to be the fps of my cam but should be ajusted to your needs
        self.font = cv.FONT_HERSHEY_SIMPLEX

    def run(self):
        started = time.time()
        while True:
            _, currentframe = self.capture.read()
            instant = time.time()  # Get timestamp o the frame
            self.processImage(currentframe)  # Process the image
            if not self.isRecording:
                if self.somethingHasMoved():
                    self.trigger_time = instant  # Update the trigger_time
                    if instant > started + 10:  # Wait 5 second after the webcam start for luminosity adjusting etc..
                        print("Something is moving !")
                        if self.doRecord:  # set isRecording=True only if we record a video
                            self.isRecording = True
                cv.drawContours(currentframe, self.currentcontours, -1, (0, 255, 0), 1)
            else:
                if instant >= self.trigger_time + 10:  # Record during 10 seconds
                    print("Stop recording")
                    self.isRecording = False
                else:
                    cv.putText(currentframe, datetime.now().strftime("%b %d, %H:%M:%S"), (25, 30), self.font, 1, 1, 2,
                               8, 0)  # Put date on the frame
                    self.writer.write(currentframe)  # Write the frame

            if self.show:
                cv.imshow("Image", currentframe)

            c = cv.waitKey(1) % 0x100
            if c == 27 or c == 10:  # Break if user enters 'Esc'.
                break

    def processImage(self, curframe):
        curframe = cv.blur(curframe, (5, 5))  # Remove false positives

        if self.absdiff_frame is None:  # For the first time put values in difference, temp and moving_average
            self.absdiff_frame = curframe
            self.previous_frame = curframe
            self.average_frame = curframe.astype(np.float32, copy=True)
        else:
            cv.accumulateWeighted(curframe, self.average_frame, 0.05)  # Compute the average

        self.previous_frame = self.average_frame.astype(np.uint8, copy=True)
        cv.absdiff(curframe, self.previous_frame, self.absdiff_frame)  # moving_average - curframe

        self.gray_frame = cv.cvtColor(self.absdiff_frame,
                                      cv.COLOR_BGR2GRAY)  # Convert to gray otherwise can't do threshold
        _, self.gray_frame = cv.threshold(self.gray_frame, 50, 255, cv.THRESH_BINARY)

        self.gray_frame = cv.dilate(self.gray_frame, kernel=(5, 5), iterations=15)  # to get object blobs
        self.gray_frame = cv.erode(self.gray_frame, kernel=(5, 5), iterations=10)

    def somethingHasMoved(self):

        # Find contours
        _, contours, _ = cv.findContours(self.gray_frame, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.currentcontours = contours  # Save contours

        for contour in contours:
            self.currentsurface += cv.contourArea(contour)

        avg = (self.currentsurface * 100) / self.surface  # Calculate the average of contour area on the total size
        self.currentsurface = 0  # Put back the current surface to 0

        if avg > self.threshold:
            return True
        else:
            return False


if __name__ == "__main__":
    detect = MotionDetectorAdaptative(doRecord=True)
    detect.run()
