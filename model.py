import time

import cv2
import dtw
import numpy as np
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def run(video_path=None, last_time=None, end_time=None):
    # initialize and select which points to analyze
    pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    # opencv video capture
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    kpts = []

    # get and use pose frame by frame
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():  # run until camera is closed
            ret, frame = cap.read()

            if not ret:
                break

            if last_time != None and end_time != None:
                now_sec = (time.time() - last_time) % 60
                if now_sec >= end_time:
                    break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB

            # get pose landmarks
            results = pose.process(rgb_frame)
            frame_kpts = dict({"kpts": [], "angle": {}})

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # calculate angle b/w left shoulder, elbow, wrist and put angle on screen
                # Define body parts for angle calculation
                joints = {
                    "left_arm": [
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_ELBOW.value,
                        mp_pose.PoseLandmark.LEFT_WRIST.value
                    ],
                    "right_arm": [
                        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                        mp_pose.PoseLandmark.RIGHT_WRIST.value
                    ],
                    "left_leg": [
                        mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.LEFT_KNEE.value,
                        mp_pose.PoseLandmark.LEFT_ANKLE.value
                    ],
                    "right_leg": [
                        mp_pose.PoseLandmark.RIGHT_HIP.value,
                        mp_pose.PoseLandmark.RIGHT_KNEE.value,
                        mp_pose.PoseLandmark.RIGHT_ANKLE.value
                    ],
                    "torso": [
                        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_HIP.value,
                        mp_pose.PoseLandmark.LEFT_KNEE.value
                    ]
                }

                for part, (a_idx, b_idx, c_idx) in joints.items():
                    a = [landmarks[a_idx].x, landmarks[a_idx].y]
                    b = [landmarks[b_idx].x, landmarks[b_idx].y]
                    c = [landmarks[c_idx].x, landmarks[c_idx].y]
                    angle = calculate_angle(a, b, c)
                    frame_kpts["angle"][part] = int(angle)

                    cv2.putText(
                        frame, f"{part}: {int(angle)}",
                        tuple(np.multiply(b, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                    )

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # draw all landmarks on the screen as small circles
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # save x and y of each keypoint and add it
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    # if i not in pose_keypoints: continue # only save keypoints that are indicated in pose_keypoints
                    pxl_x = int(round(landmark.x * frame.shape[1]))
                    pxl_y = int(round(landmark.y * frame.shape[0]))
                    cv2.circle(frame, (pxl_x, pxl_y), 3, (0, 0, 255), -1)
                    curr_kpt = [pxl_x, pxl_y]
                    frame_kpts["kpts"].append(curr_kpt)

            else:
                frame_kpts = [[-1, -1]] * len(pose_keypoints)  # fill data with [-1,-1] if no person is present

            kpts.append(frame_kpts)  # add this frame to collection of all frames
            cv2.imshow('Pose Landmarks', frame)  # display output frame

    cv2.destroyAllWindows()
    cap.release()
    return kpts


def preprocess_data(video_data):
    processed_data = []
    for frame in video_data:
        if isinstance(frame, dict) and 'kpts' in frame:
            processed_data.append(np.array(frame['kpts']).flatten())
    return np.array(processed_data)


def align_videos(video1, video2):
    video1_kpts = preprocess_data(video1)
    video2_kpts = preprocess_data(video2)
    dist, cost, acc_cost, path = dtw.accelerated_dtw(video1_kpts, video2_kpts, dist='euclidean')
    aligned_video1 = [video1[i] for i in path[0]]
    aligned_video2 = [video2[j] for j in path[1]]
    return aligned_video1, aligned_video2, path


def normalize_values(arr):
    arr = np.array(arr)
    arr[arr < 0.9] = 0  # Set values below 0.9 to 0
    mask = arr >= 0.9  # Mask values that need normalization
    arr[mask] = (arr[mask] - 0.9) / 0.1  # Normalize 0.9-1.0 range to 0.0-1.0
    return arr


def similarity_preprocesser(video_aligned):
    video_1_processed = []
    for frame in video_aligned:
        if isinstance(frame, dict) and 'kpts' in frame:
            curr = np.array(frame['kpts']).flatten().tolist()
            for key in frame['angle']:
                curr.append(frame['angle'][key])
            video_1_processed.append(curr)
    return np.array(video_1_processed)


def find_low_score_segments(scores, threshold=0.85, min_length=20, fps=20):
    segments = []
    start_index = None
    count = 0

    for i in range(len(scores)):
        if scores[i] < threshold:
            if start_index is None:  # Start of a new segment
                start_index = i
            count += 1
        else:
            if start_index is not None:  # End of a segment
                if count >= min_length:
                    segments.append((start_index / fps, (i - 1) / fps))
                start_index = None
                count = 0

    # Check if the last segment ends before the end of the array
    if start_index is not None and count >= min_length:
        segments.append((start_index / fps, (len(scores) - 1) / fps))

    return segments


def save_overlay_video(video1_path, video2_path, scores, path):
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    frame_count = len(scores)
    frame_width = int(video1.get(3))
    frame_height = int(video1.get(4))
    output_filename = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 20
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width * 2, frame_height))

    for i in range(frame_count):
        score = scores[i]
        video1.set(cv2.CAP_PROP_POS_FRAMES, path[0][i])
        video2.set(cv2.CAP_PROP_POS_FRAMES, path[1][i])
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()
        if not ret1 or not ret2:
            print("Error reading frame")
            break

        combined_frame = np.hstack((frame1, frame2))
        if (score > 0.85):
            cv2.putText(combined_frame, f"Score: {score:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(combined_frame, f"Score: {score:.2f}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(combined_frame)
        cv2.imshow("Frame", combined_frame)

    video1.release()
    video2.release()
    out.release()
    cv2.destroyAllWindows()


def save_video_segments(video_path, low_score_segments, dtw_path, fps=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or fps
    duration = frame_count / frame_rate

    mapped_segments = [(dtw_path[0][int(start * fps)], dtw_path[0][int(end * fps)]) for start, end in
                       low_score_segments]
    all_segments = []
    last_end = 0

    for start_frame, end_frame in mapped_segments:
        if end_frame - start_frame >= fps:
            start_time = start_frame / frame_rate
            end_time = end_frame / frame_rate
            if last_end < start_time and (start_time * fps) - (last_end * fps) >= fps:
                all_segments.append((last_end, start_time, "good"))
            all_segments.append((start_time, end_time, "bad"))
            last_end = end_time

    if last_end < duration and (duration * fps) - (last_end * fps) >= fps:
        all_segments.append((last_end, duration, "good"))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = None
    label = None
    frame_idx = 0
    segment_idx = 0

    for start, end, new_label in all_segments:
        if label != new_label:
            if out:
                out.release()
            out = cv2.VideoWriter(f"{video_path}_{segment_idx}_{new_label}.mp4", fourcc, frame_rate,
                                  (frame_width, frame_height))
            segment_idx += 1
            label = new_label

        while frame_idx < end * fps:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= start * fps:
                out.write(frame)
            frame_idx += 1

    if out:
        out.release()

    cap.release()
    cv2.destroyAllWindows()


def process(videopath_1, videopath_2):
    kpts_vid1 = run(video_path=videopath_1)
    kpts_vid2 = run(video_path=videopath_2)
    aligned_frames1, aligned_frames2, path = align_videos(kpts_vid1, kpts_vid2)
    video_1_processed = similarity_preprocesser(aligned_frames1)
    video_2_processed = similarity_preprocesser(aligned_frames2)
    frame_scores = np.diag(cosine_similarity(video_1_processed, video_2_processed))
    scores = normalize_values(frame_scores / np.max(frame_scores))
    save_overlay_video(videopath_1, videopath_2, scores, path)
'''low_score_segments = find_low_score_segments(scores)
    save_video_segments(videopath_1, low_score_segments, path)
    save_video_segments(videopath_2, low_score_segments, path)'''