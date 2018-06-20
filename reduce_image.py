import cv2

cap = cv2.VideoCapture('./12.avi')
i = 0
while(True):
	i+= 1
	ret, frame = cap.read()

	if ret == False:
		break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if i >= 40 and i <= 300:
		frame = cv2.resize(frame, (0,0), fx = 0.1, fy = 0.1)
		cv2.imshow('frame',frame)
		filename = "./12_" + str(i) + ".png"
		print(filename)
		
		cv2.imwrite(filename, frame)

cap.release()