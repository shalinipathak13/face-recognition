import numpy as np
import cv2

#instantiate a came object to capture images
cam = cv2.VideoCapture(0)
# create a haar-cascade object for face recognition/detection
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

#create a place holderfor storing the data
data = []
ix = 0 # current frame number

while True:
	#retrieve the ret(boolean) and frame from camera
	ret, frame = cam.read()

	#if the camera is working fine, we proceed to extrac the face
	if ret == True:
		# convert the current frame to grayscale
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		#apply the haar cascade to detect faces in the current frame
		#the other parameters 1.3 and 5 are the fine tuning parameters
		#for the haar cascade object 
		faces = face_cas.detectMultiScale(gray, 1.3, 5)

		#for each face object we get, we have
		#the corner coords (x,y)
		#and the width and height of the face

		for (x, y, w, h) in faces:

			#get the face component from the image frame
			face_component = frame[y:y+h, x:x+w, :]

			#resize the face image to 50x50x3 
			fc = cv2.resize(face_component, (50,50))


			# store the face data after every 10frames
			#only if the number of entries is less than 20

			if ix%10 == 0 and len(data) < 20:
				data.append(fc)

			#for visualization, draw a rectangle around the face
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
		ix += 1 # increment the current frame number 
		cv2.imshow('frame', frame) #display the frame

		#if the user presses the escape key(ID:27)
		#or the number of images hits 20, we stop recording
		if cv2.waitKey(1) == 27 or len(data) >= 20:
			break
	else:
		#if camera is not working , print error
		print("error")

#now we destroy the windows we have created
cv2.destroyAllWindows()

#convert the data to a numpy format
data = np.asarray(data)

#print the shape as a sanity-check
print(data.shape)

#save the data as numpy matrix in an encoded format
np.save('face_03',data)

#we'll run the script for different people and store the 
#data into multiple files
#

