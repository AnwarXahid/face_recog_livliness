import cv2
import time


start_time = time.time()
cap = cv2.VideoCapture(0)
print("Hello Khashi")
print(time.time() - start_time)
#
#
# cap = cv2.VideoCapture(0)
#
#
# start = time.time()
# frame_count = 0
#
# while(True):
#     # Capture frame-by-frame
#     beforeCapture = time.time()
#     ret, frame = cap.read()
#     frame_count += 1
#
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('face detected!',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     afterCapture = time.time()
#     print(afterCapture-beforeCapture)
#
# time_taken = time.time() -start
# print("frame_count             = {}".format(frame_count))
# print("total time             = {}".format(time_taken))
# print("frame_per_sec             = {}".format(frame_count/time_taken))
# start = time.time()
# # your code
# end = time.time()
# time_taken = end - start
# print('Time: ', time_taken)