#!/usr/bin/env python3
import numpy as np
import cv2, numpy as np
kalman = cv2.KalmanFilter(3,3,3,cv2.CV_64F)		#Declare 3 dynamical parameter 3 state parameter 3 control parameter
kalman.transitionMatrix=1.0*np.eye(3)			#Transition set to identity only the control and the noise are changing the state
#Dummy measurement matrix,  3 measurement each times !
kalman.measurementMatrix=np.matrix([[1.0,2.2,3.10],
				    [1.1,2.2,3.30],
				    [1.2,2.4,3.70],
				    [1.3,2.430,3.50],
				    [1.5,2.50,3.340],
				    [1.5,2.60,3.70],
				    [1.7,2.0,3.30],
				    [1.9,2.30,3.20],
				    [1.0,2.40,3.50],
				    [1.3,2.60,3.60],
				    [1.4,2.20,3.20],
				    [1.6,2.0,3.20]])

kalman.measurementNoiseCov=1.1*np.eye(3)
kalman.processNoiseCov=1.2*np.eye(3)
kalman.controlMatrix=1.0*np.eye(3)

uk=np.matrix([[0.0],
		[1.2],
		[1.4]])

print("Now checking")
print("Measurement Matrix")
print(kalman.measurementMatrix)
print("Transition Matrix")
print(kalman.transitionMatrix)
print("MeasurementNoiseCov")
print(kalman.measurementNoiseCov)
print("ProcessNoiseCov")
print(kalman.processNoiseCov)
print("ControlMatrix")
print(kalman.controlMatrix)

print("Now attempted to make use of the underdocumented kalman filter in opencv with python ;)")
predicted=kalman.predict(uk)
print("Predicted:")
print(predicted)
print("Will (try to) correct by feeding directly the predicted to correction")
print("So here OpenCV is telling me that KF can not correct its own prediction even though the KF here has same number of dynamical,state and control parameters ?\n...")
estimated=kalman.correct(predicted)

