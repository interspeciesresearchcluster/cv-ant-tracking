import cv2
import numpy as np

# CAPTURE
use_video = False  # Set to False to use the camera

if use_video:
    # Path to your video file
    video_path = '4kVideoAnts.mp4'  # Update this path to your video file
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)  # Use the camera

if not cap.isOpened():
    print("Error: Could not open camera or video file.")
    exit()

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840) # 4k
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {width}x{height}")
if not cap.isOpened():
    print("Error: Could not open camera or video file.")
    exit()

# Read the first frame from the video
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

# Resize window for better visualization
cv2.namedWindow("Select Corners", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Corners", 1280, 720)

# Store clicked points
selected_points = []

# Mouse click callback function
def select_points(event, x, y, flags, param):
    global selected_points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            selected_points.append((x, y))
            print(f"Point {len(selected_points)}: {x}, {y}")

        # Draw the selected points on the frame
        for i, (px, py) in enumerate(selected_points):
            cv2.circle(frame, (px, py), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"{i+1}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # If 4 points are selected, draw a polygon
        if len(selected_points) == 4:
            cv2.polylines(frame, [np.array(selected_points)], isClosed=True, color=(255, 0, 0), thickness=2)
            selected_points = [selected_points[0], selected_points[1], selected_points[3], selected_points[2]]

        cv2.imshow("Select Corners", frame)

# Set the mouse callback function
cv2.setMouseCallback("Select Corners", select_points)

while True:
    cv2.imshow("Select Corners", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):  # Reset selection
        selected_points = []
        ret, frame = cap.read()  # Reload original frame
        if not ret:
            print("Error: Could not reload frame.")
            break
        print("Reset points! Click again.")

    elif key == ord('q'):  # Quit and save points
        if len(selected_points) == 4:
            print("Final selected points (for perspective transform):")
            print(selected_points)
        else:
            print("Not enough points selected! Select 4 points before quitting.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
