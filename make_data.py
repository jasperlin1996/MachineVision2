import cv2
import numpy as np

cap = cv2.VideoCapture('test01.avi')
bgSubThreshold = 100
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
imgSkin = np.zeros((int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3), np.uint8)
fourcc = cv2.VideoWriter_fourcc(*'XVID')		# w  ,h
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (340,340))
recording = 0

output = np.random.uniform(0,0,size=(340,340,3))

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

	output[:,:,0] = im2[70:410,150:490]
	output[:,:,1] = im2[70:410,150:490]
	output[:,:,2] = im2[70:410,150:490]
	print(im2)
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
#	cv2.line(output,(150,0),(150,480),(255,0,0),5)
#	cv2.line(output,(640-150,0),(640-150,480),(255,0,0),5)
#	cv2.line(output,(150,70),(640-150,70),(255,0,0),5)
#	cv2.line(output,(150,480-70),(640-150,480-70),(255,0,0),5)
	#cv2.rectangle(output, )
	
	frame = frame[:,::-1,:]

	cv2.imshow('webcam', output)
	print(im2.shape)
	if cv2.waitKey(1) & 0xFF == ord('r'):
		recording = 1
		print("------Recording------")
	if recording == 1:
		o = cv2.cvtColor(im2[70:410,150:490],cv2.COLOR_GRAY2BGR)
		out.write(o)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		print("--------Exit---------")
		break

cap.release()
out.release()
cv2.destroyAllWindows()