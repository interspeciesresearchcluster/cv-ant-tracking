import cv2
import numpy as np


# Define the resolution
projection_mapping_target_resolution = (1920, 1080)

# Initialize rectangle corner points (centered)
projection_mapping_target_points = np.array([[790, 92], [1804, 138], [1846, 970], [790, 1015]], dtype=np.int32)

# Track which corner is being dragged (-1 means none)
dragging_corner = -1
update_needed = True  # Flag to update the display only when needed

# Create a black canvas
canvas = np.zeros((projection_mapping_target_resolution[1], projection_mapping_target_resolution[0], 3), dtype=np.uint8)

# Mouse callback function
def update_rectangle(event, x, y, flags, param):
    global dragging_corner, projection_mapping_target_points, update_needed

    if event == cv2.EVENT_LBUTTONDOWN:
        # Find the closest corner to the mouse click
        distances = [np.linalg.norm((px - x, py - y)) for px, py in projection_mapping_target_points]
        dragging_corner = np.argmin(distances)

    elif event == cv2.EVENT_MOUSEMOVE and dragging_corner != -1:
        # If dragging, update the corner position **only if it changed**
        if (projection_mapping_target_points[dragging_corner][0] != x or
                projection_mapping_target_points[dragging_corner][1] != y):
            projection_mapping_target_points[dragging_corner] = (x, y)
            update_needed = True  # Mark display for update

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_corner = -1  # Stop dragging

# Initialize opencv window
cv2.namedWindow('Main Window', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Main Window',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Main Window", update_rectangle)

while True:
    if update_needed:
        # Reset frame only when needed
        frame = canvas.copy()

        # Draw the white rectangle
        cv2.polylines(frame, [projection_mapping_target_points], isClosed=True, color=(255, 255, 255), thickness=3)
        cv2.fillPoly(frame, [projection_mapping_target_points], (255, 255, 255))

        # Draw draggable corners
        for x, y in projection_mapping_target_points:
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        # Show the window (only when necessary)
        cv2.imshow("Main Window", frame)
        update_needed = False  # Reset update flag

    # Key press logic
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Print final coordinates and exit
        print("Final projection_mapping_target_points:")
        projection_mapping_target_points_list = projection_mapping_target_points.tolist()
        projection_mapping_target_points_list = [projection_mapping_target_points_list[0], projection_mapping_target_points_list[1], projection_mapping_target_points_list[3], projection_mapping_target_points_list[2]]
        print(projection_mapping_target_points_list)
        break

cv2.destroyAllWindows()
