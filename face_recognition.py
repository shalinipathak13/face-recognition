import numpy as np 
import cv2

cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

f1 = np.load('./face_01.npy').shape[0]
f2 = np.load('./face_02.npy').shape[0]
f3 = np.load('./face_03.npy').shape[0]
print(f1,f2,f3)

f_01 = np.load('./face_01.npy').reshape(f1,50*50*3)
f_02 = np.load('./face_02.npy').reshape(f2,50*50*3)
f_03 = np.load('./face_03.npy').reshape(f3,50*50*3)

print (f_01.shape , f_02.shape, f_03.shape)

names = {
	0: 'Shalini',
	1: 'Shruti',
	2: 'Bina',
}

labels = np.zeros((f1+f2+f3,1))

labels[:f1,:] = 0.0
labels[f1:f1+f2,:] = 1.0
labels[f1+f2:,:] = 2.0

print(labels)
data = np.concatenate([f_01,f_02,f_03])
print (data.shape, labels.shape)

def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, targets, k=5):
    m = train.shape[0]
    dist = []
    for ix in range(m):
        # compute distance from each point and store in dist
        dist.append(distance(x, train[ix]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]

while True:

	ret, frame = cam.read()

	
	if ret == True:
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cas.detectMultiScale(gray, 1.3, 5)

		for (x, y, w, h) in faces:

			face_component = frame[y:y+h, x:x+w, :]
			fc = cv2.resize(face_component, (50,50))

			lab = knn(fc.flatten(), data, labels)
			text = names[int(lab)]
			cv2.putText(frame,text,(x,y), font, 1, (225, 225, 0), 2)

			cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 225), 2)
		cv2.imshow('frame',frame)
		
		if 	cv2.waitKey(1) == 27:
			break

	else:
		print('Error')

cv2.destroyAllWindows()

