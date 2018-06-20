import cv2
import numpy as np
import h5py
from keras.models import load_model

cap = cv2.VideoCapture('D:\\HW2\\test1.avi')
bgSubThreshold = 100
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
imgSkin = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3), np.uint8)

model = load_model('test_model.h5')

result_char = ['子','丑','寅','卯','辰','巳','午','未','申','酉','戌','亥']

def decode_fourcc(v):
	v = int(v)
	return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

def inner_product(p1, p2, p3):
	v1 = np.array([p2[0,0]-p1[0,0], p2[0,1]-p1[0,1]])
	v2 = np.array([p2[0,0]-p3[0,0], p2[0,1]-p3[0,1]])
	return (np.dot(v1,v2)/(np.sqrt(v1[0]**2+v1[1]**2)*np.sqrt(v2[0]**2+v2[1]**2)))

j = 0
while(True):
	j += 1
	ret, frame = cap.read()

	
	blur = cv2.GaussianBlur(frame, (31,31), 0)

	ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCR_CB)
	

	# video info
	fourcc = cap.get(cv2.CAP_PROP_FOURCC)
	codec = decode_fourcc(fourcc)
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
	imgSkin = frame.copy()

	skin = cv2.inRange(ycrcb, np.array([20, 155-22, 120-22]), np.array([255, 155+22, 120+22]))
	im2, contours, hierarchy = cv2.findContours(skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	
	for cnt in contours:
		hull = cv2.convexHull(cnt)
		moment = cv2.moments(hull)
		cv2.drawContours(frame, [hull], 0, (0,0,255), 2)
		if moment["m00"] != 0:
			x = int(moment["m10"]/moment["m00"])
			y = int(moment["m01"]/moment["m00"])
			cv2.circle(frame, (x,y), 5, (0,0,255), 5)
			maxDistance = 0
			maxFinger = (0,0)
			i = 12
			#print(cnt.shape)
			if cnt.shape[0] > 12:
				angle = np.array([cnt[0],cnt[6],cnt[12]])
				for point in cnt:

					if ((point[0,0]-x)**2 + (point[0,1]-y)**2) > maxDistance:
						maxDistance = ((point[0,0]-x)**2 + (point[0,1]-y)**2)
						maxFinger = (point[0,0], point[0,1])
						cv2.circle(frame, maxFinger, 5, (255,0,0), 5)
	#				
	#				i += 1
	#				if i % 6 == 0:
	#					angle = angle[1:3]
	#					angle = np.append(angle,np.array([point]), axis = 0)
	#					#print(angle[1].shape)
	#					if inner_product(angle[0],angle[1],angle[2]) > -0.5:
	#						cv2.circle(frame, (angle[1,0,0], angle[1,0,1]), 5,(255,0,0),5)


	cv2.drawContours(frame, contours, -1, (0,255,0) ,3)
#	cv2.imshow('webcam', frame)
	im2 = frame[:,:,0]
	im2 = cv2.resize(im2,(0,0),fx = 0.1, fy = 0.1)
	cv2.imshow('check', im2)
	im3 = np.reshape(im2, im2.shape[0]*im2.shape[1])
#	print(im2.shape)
#	print(im2)
	if j % 30 == 0:
		ans = model.predict(np.array([im3]))
		print(ans)
		max = -1
		for _ in ans:
			for index,value in enumerate(_):
				if value > max:
					max = value
		for _ in ans:
			for index,value in enumerate(_):
				if value == max:
					if value > 0.5:

						print("we got : {}".format(result_char[index]))
					else:
						print("we got nothing")

		#cv2.imshow('test_case', im2)
		pass


	#print("Video size: {} x {}".format(width, height))
	#print("Codec: " + codec)
	#

	
	frame = frame[:,::-1,:]
	cv2.imshow('webcam', frame)
	
	if cv2.waitKey(25) & 0xFF == ord('q'):
		break

cap.release()

cv2.destroyAllWindows()