import cv2
import time
import numpy as np
import os

# GENERAL VARIABLES
start_time = time.time()
frame_count = 0
frame_rate = 10
save_interval = 1/frame_rate  # Save an image every 0.1 seconds

sandbox_aspect_ratio = 0.793103448 # Real sandbox internal dimensions = 149.5cm x 188.5cm
sandbox_frame_resolution = 1280

sandbox_bounds = [(1476, 391), (2562, 403), (1272, 1744), (2706, 1795)]

projection_mapping_target_points = [[762, 57], [1819, 105], [768, 1040], [1867, 983]]
projection_mapping_target_resolution = (1920, 1080)

# ANT TRACKING VARIABLES
ant_paths = {}  # Tracks paths for active ants (ant_id -> list of (x, y))
ant_trackers = {}  # Tracks the last known position of each ant (ant_id -> (x, y))
inactive_frames = {}  # Tracks how many frames an ant has been inactive (ant_id -> count)
appearance_frames = {}  # Tracks how many frames a candidate ant has appeared consecutively
next_ant_id = 0  # Counter for assigning unique IDs to ants
max_inactive_frames = 12  # Maximum frames an ant can be inactive
min_appearance_frames = 2  # Minimum frames for a candidate ant to become valid
bake_frames = min_appearance_frames*3 # The number of frames after which the ant path will become permanently baked
distance_threshold = 15 #Maximum pixel distance between frames before a path is broken
ant_size_min = 1 # The pixel size of the contour that is minimum to be an ant
ant_size_max = 50 # The pixel size of the contour that is maximum to be an ant

# CAPTURE
use_video = True  # Set to False to use the camera
loopVideo = True

if use_video:
    # Path to your video file
    video_path = 'data/example.mp4'  # Update this path to your video file
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

# Initialize opencv window
cv2.namedWindow('Main Window', cv2.WINDOW_NORMAL)
# cv2.setWindowProperty('Main Window',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize background subtractor for tracking
back_sub = cv2.createBackgroundSubtractorKNN()

# Initialise the ant bake image
ant_bake = np.zeros((sandbox_frame_resolution, int(sandbox_frame_resolution*sandbox_aspect_ratio), 3), dtype=np.uint8)
#ant_bake = np.full((sandbox_frame_resolution, int(sandbox_frame_resolution * sandbox_aspect_ratio), 3), 255, dtype=np.uint8)
#ant_bake = np.ones((sandbox_frame_resolution, int(sandbox_frame_resolution * sandbox_aspect_ratio), 3), dtype=np.uint8) *50

# Define the path to your SSD (D: drive) and the "AntsStranding" folder
# save_path = "D:/AntsStranding/Recordings"  # Update this to the correct SSD path

# Ensure the directories exist
timestamp_start = time.strftime("%Y%m%d_%H%M%S")
# os.makedirs(f"{save_path}/{timestamp_start}/ants", exist_ok=True)
# os.makedirs(f"{save_path}/{timestamp_start}/camera", exist_ok=True)

print("Initializing...")
time.sleep(2)  # 2-second delay for initialization
print("Starting recording and tracking...")

# Function to correct perspective of rectangle
def correct_sandbox_perspective(img):
    final_width = int(sandbox_frame_resolution * sandbox_aspect_ratio)
    final_height = sandbox_frame_resolution
    # Size of the Transformed Image
    pts1 = np.float32(sandbox_bounds)
    pts2 = np.float32([[0, 0], [final_width, 0], [0, final_height], [final_width, final_height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (final_width, final_height))
    return dst

# Function to map the video frame to 4 points and display on a black background
def project_mapping(frame):

    # Resize the input frame to the target resolution
    frame_resized = cv2.resize(frame, projection_mapping_target_resolution)

    # Create a black background with the target resolution
    black_background = np.zeros((projection_mapping_target_resolution[1], projection_mapping_target_resolution[0], 3), dtype=np.uint8)

    # Rotate the resized frame by 90 degrees
    frame_rotated = cv2.rotate(frame_resized, cv2.ROTATE_90_CLOCKWISE)

    # Define the source points (corners of the frame)
    src_h, src_w = frame_rotated.shape[:2]
    src_points = np.float32([[0, 0], [src_w, 0], [0, src_h], [src_w, src_h]])

    # Define the target points (ensure they fit within the target resolution)
    dst_points = np.float32(projection_mapping_target_points)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Warp the perspective of the rotated frame to match the target points
    warped_frame = cv2.warpPerspective(frame_rotated, M, projection_mapping_target_resolution)

    # Overlay the warped frame onto the black background
    black_background = cv2.addWeighted(black_background, 1, warped_frame, 1, 0)

    return black_background
def detect_candidate_ants(input_image):
    global candidate_ants, ant_size_min, ant_size_max
    contours, _ = cv2.findContours(input_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect candidate ants based on contours
    candidate_ants = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if ant_size_min <= area <= ant_size_max:  # Filter ants by contour area
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                # Calculare centroids
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                candidate_ants.append((cx, cy))
                print(str(area) + " Ant Size" ) # debug um herausfinden wie groß eine ameise ist

    print(f"Candidate Ants Detected: {len(candidate_ants)}")
    return candidate_ants
def track_ants(candidate_ants):
    global ant_trackers, ant_paths, next_ant_id, inactive_frames, appearance_frames, distance_threshold
    # Track and validate ants
    updated_ids = set()
    for cx, cy in candidate_ants:
        # Match candidate ants to existing ants
        matched = False
        for ant_id, last_position in ant_trackers.items():
            distance = np.linalg.norm(np.array(last_position) - np.array((cx, cy)))
            if distance < distance_threshold:  # Match found
                ant_trackers[ant_id] = (cx, cy)
                ant_paths[ant_id].append((cx, cy))
                updated_ids.add(ant_id)
                inactive_frames[ant_id] = 0  # Reset inactivity counter
                print(str(distance) + "Distance")   # debug um herauszufinden wie weit eine ameise sich bewegt bei 3 fps
                matched = True
                break

        # If no match found, track as a new candidate
        if not matched:
            if (cx, cy) not in appearance_frames:
                appearance_frames[(cx, cy)] = 0
            appearance_frames[(cx, cy)] += 1
            print("no Match")


            # If candidate ant appears for enough frames, assign it an ID
            if appearance_frames[(cx, cy)] >= min_appearance_frames:
                ant_trackers[next_ant_id] = (cx, cy)
                ant_paths[next_ant_id] = [(cx, cy)]
                inactive_frames[next_ant_id] = 0
                updated_ids.add(next_ant_id)
                next_ant_id += 1
                del appearance_frames[(cx, cy)]  # Remove from candidate list

    # Update inactivity counters for ants not updated
    for ant_id in list(ant_trackers.keys()):
        if ant_id not in updated_ids:
            inactive_frames[ant_id] += 1
        else:
            inactive_frames[ant_id] = 0
    

    # Remove ants that have been inactive too long
    for ant_id, count in list(inactive_frames.items()):
        if count > max_inactive_frames:
            del ant_trackers[ant_id]
            del inactive_frames[ant_id]
            del ant_paths[ant_id]

    #print(f"Ant Paths Detected: {len(ant_paths)}")
    return ant_paths
def visualise_ants():
    global ant_bake, min_appearance_frames, bake_frames
    

    # Bake ant paths only for ants that have met the minimum appearance requirement
    for ant_id, path in ant_paths.items():
        if len(path) >= bake_frames:  # Only process if the ant has appeared enough frames
            pathLen = len(path)
            if pathLen > bake_frames +1:
                for i in range(1, len(path)):
                    if path[i - 1] and path[i]:
                        cv2.line(ant_bake, path[i - 1], path[i], (0, 255, 0), 1)
    
    image = ant_bake.copy()
    
    # Draw paths for valid ants (those with enough appearance frames) # here is a mistake because also ants with big paths meaning noise get shown 
    for ant_id, path in ant_paths.items():
        if len(path) >= min_appearance_frames:  # Only draw if minimum appearance threshold is met
            for i in range(1, len(path)):
                if path[i - 1] and path[i]:
                    cv2.line(image, path[i - 1], path[i], (0, 255, 0), 1)

    return image

# Main loop for capture and tracking
while True:
    ret, frame = cap.read()
    if not ret:
        if loopVideo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
            print("Error: Failed to capture frame or end of video reached.")
            break

    if time.time() - start_time >= frame_count * save_interval:
        print(f"Processing frame {frame_count}")

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        perspectived = correct_sandbox_perspective(gray)

        # give buffer in the beginning to start the programm and open frame
        if frame_count < 40:
            # Create a full black image with the same shape as 'perspectived'
            # Apply background subtraction
            backsub_image = back_sub.apply(perspectived)#can this be comnmented out?
            backsub_image = np.zeros_like(perspectived)
            #backsub_image = np.full_like(perspectived, 255)

        else:
            # Apply background subtraction
            backsub_image = back_sub.apply(perspectived)

        # Morphological operations to reduce noise
        kernel = np.ones((2, 2), np.uint8)
        backsub_image = cv2.erode(backsub_image, kernel, iterations=1)
        backsub_image = cv2.dilate(backsub_image, kernel, iterations=1)

        # Convert fg_mask to a binary image (black and white only)
        _, back_sub_binary = cv2.threshold(backsub_image, 1, 255, cv2.THRESH_BINARY)

        # Count the white pixels in back_sub_binary (assuming it's a binary image where white is 255)
        white_pixels = cv2.countNonZero(back_sub_binary)
        print(str(white_pixels) + " white pixels")
        # If there are more than 40 white pixels, substitute back_sub_binary with a full black image of the same size
        if white_pixels > 300:
            back_sub_binary = np.zeros_like(back_sub_binary)
        # Now detect candidate ants using the possibly modified back_sub_binary
            
        candidate_ants = detect_candidate_ants(back_sub_binary)
        track_ants(candidate_ants)
        ants = visualise_ants()
    
        #Uncomment when you want to debug 
        # BW to RGB for overlaying
        back_sub_colored = cv2.cvtColor(back_sub_binary, cv2.COLOR_GRAY2BGR)
        perspectived_colored = cv2.cvtColor(perspectived, cv2.COLOR_GRAY2BGR)

        #Important parameter to change live maske it 0 when projecting
        overlay = cv2.addWeighted(back_sub_colored, 0.5, perspectived_colored, 0.1, 0) # For Debugging
        overlay = cv2.addWeighted(overlay, 1.0, ants, 1.0, 0)
    
        output = project_mapping(overlay)

        # Save the image "ants" and "perspectived" with the current framecount and timecode
        # current_time =  time.strftime("%Y%m%d_%H%M%S")  # Get the current time once
        # ants_filename = f"Recordings/{timestamp_start}/ants/frame_{frame_count}_{current_time}.png"
        # perspectived_filename = f"Recordings/{timestamp_start}/camera/frame_{frame_count}_{current_time}.png"
        # if frame_count % 1 == 0:
        #     cv2.imwrite(ants_filename, ants)
        #     cv2.imwrite(perspectived_filename, perspectived)

        if frame_count % 1 == 0:
            cv2.imshow('Main Window', output)

        frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Recording stopped early.")
        break

cap.release()
cv2.destroyAllWindows()

#adjust variables
#add saving the images to later on see how they walked. Maybe save ant_id and tie it to frame count to recalculate the image
#add updating the paths every third frame. 
#add when to much white appears frame is invalid.