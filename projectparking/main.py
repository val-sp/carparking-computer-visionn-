import cv2
import matplotlib.pyplot as plt
import numpy as np
from util import get_parking_spots_bboxes, empty_or_not


def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))


# Paths
mask_path = 'C:/Users/Shubhrojit Panda/Desktop/dataparking/mask_1920_1080.png'
video_path = 'C:/Users/Shubhrojit Panda/Desktop/dataparking/data/parking_1920_1080.mp4'

# Load mask
mask = cv2.imread(mask_path, 0)
cap = cv2.VideoCapture(video_path)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Initialize variables
spots_status = [None for _ in spots]
diffs = [None for _ in spots]
previous_frame = None
frame_nmr = 0
step = 30
ret = True

# Setup matplotlib for real-time plotting
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
bar = ax.bar(['Available Spots', 'Total Spots'], [0, len(spots)], color=['green', 'blue'])
ax.set_ylim(0, len(spots))  # Set y-limit to the total number of parking spots

while ret:
    ret, frame = cap.read()

    if frame_nmr % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

    if frame_nmr % step == 0:
        if previous_frame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]

        for spot_indx in arr_:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

        # Update the bar chart with the current number of available spots
        bar[0].set_height(sum(spots_status))
        plt.draw()
        plt.pause(0.01)  # Small pause to update the plot

    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    for spot_indx, spot in enumerate(spots):
        spot_status = spots_status[spot_indx]
        x1, y1, w, h = spots[spot_indx]

        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_nmr += 1

cap.release()
cv2.destroyAllWindows()
plt.ioff()  # Disable interactive mode
plt.show()  # Display final plot after the loop
