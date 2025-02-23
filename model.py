import time
import os
import cv2
# import dtw
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cdist
from moviepy.editor import ImageSequenceClip
from sklearn.metrics.pairwise import cosine_similarity
import re


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
            # cv2.imshow('Pose Landmarks', frame)  # display output frame

    cv2.destroyAllWindows()
    cap.release()
    return kpts

def preprocess_data(video_data):
    processed_data = []
    for frame in video_data:
        if isinstance(frame, dict) and 'kpts' in frame:
            processed_data.append(np.array(frame['kpts']).flatten())
    return np.array(processed_data)

def accelerated_dtw(x, y, dist, warp=1):
    if np.ndim(x) == 1:
        x = x.reshape(-1, 1)
    if np.ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

def align_videos(video1, video2):
    video1_kpts = preprocess_data(video1)
    video2_kpts = preprocess_data(video2)
    dist, cost, acc_cost, path = accelerated_dtw(video1_kpts, video2_kpts, dist='euclidean')
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

def new_save_video(video1_path, video2_path, scores, path):
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)
    output_dir = os.path.join("/static/uploads", "frames/")
    frames = []

    for i in range(len(scores)):
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

        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        combined_frame = np.array(combined_frame)
        sentiment = "good" if score > 0.85 else "bad"

        from PIL import Image
        output_filename = os.path.join("static/uploads/frames/", f"{i}_output_{sentiment}.jpeg")
        im = Image.fromarray(combined_frame)
        im.save(output_filename)
        frames.append(combined_frame)

    clip = ImageSequenceClip(frames, fps=20)
    output_filename = os.path.join("static/uploads/", "output.mp4")
    clip.write_videofile(output_filename, codec='libx264')

    video1.release()
    video2.release()
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


import google.generativeai as genai1
from dotenv import load_dotenv
load_dotenv()
import os
from google.ai.generativelanguage_v1beta.types import content
from google import genai
import json

gemini_api_key = os.getenv('GEMINI_API_KEY')


def wait_for_files_active(files, max_retries=10, initial_delay=5):
    """Waits for the given files to be active using exponential backoff.

    This avoids constant polling by gradually increasing the delay between retries.
    """
    print("Waiting for file processing...")
    for file in files:
        retries = 0
        delay = initial_delay
        while retries < max_retries:
            file_state = genai1.get_file(file.name).state.name
            if file_state == "ACTIVE":
                print(f"File {file.name} is now active.")
                break
            elif file_state != "PROCESSING":
                raise Exception(f"File {file.name} failed to process with state: {file_state}")

            print(f"File {file.name} still processing. Retrying in {delay} seconds...")
            time.sleep(delay)
            retries += 1
            delay *= 2  # Exponential backoff

        if retries == max_retries:
            raise TimeoutError(f"File {file.name} failed to become active after {max_retries} retries.")

    print("All files are active.")


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


def generate_feedback_with_frames(similarity_matrix, body_parts, num_low_frames=3):
    feedback = []

    for i, part in enumerate(body_parts):
        body_part_scores = similarity_matrix[:, i]
        low_frames = np.argsort(body_part_scores)[:num_low_frames].tolist()
        timestamps = frames_to_timestamp(low_frames) if low_frames else ["N/A"]
        avg_score = round(float(np.mean(body_part_scores)), 2)

        if avg_score > 0.85:
            comment = "Great alignment!"
        elif avg_score > 0.7:
            comment = "Good, but slight adjustments needed."
        else:
            comment = "Needs improvement in movement or positioning."

        feedback.append({
            "body_part": part,
            "low_similarity_frames": low_frames,
            "timestamp": timestamps[0] if low_frames else "N/A",
            "average_similarity_score": avg_score,
            "feedback": comment,
            "summary": f"{part.replace('_', ' ').title()}: {comment}"
        })

    return feedback


def upload_to_gemini(path, mime_type=None):
    file = genai1.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def frames_to_timestamp(frame_indices, fps=30):
    """Convert frame indices to hh:mm:ss:ms timestamps."""
    timestamps = []
    for frame in frame_indices:
        total_seconds = frame / fps
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        milliseconds = int((total_seconds - int(total_seconds)) * 1000)
        timestamps.append(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}:{milliseconds:03d}")
    return timestamps

def detect_bad_frame_sections(frames_folder):
    """
    Scan a directory for bad frames and detect sections with 8+ consecutive bad frames.

    :param frames_folder: Path to the folder containing frames
    :return: JSON object with bad frame sections
    """
    frame_pattern = re.compile(r"(\d+)_output_bad\.jpeg")
    frames = []

    # Scan directory and extract bad frames
    for filename in sorted(os.listdir(frames_folder), key=lambda x: int(re.search(r"(\d+)", x).group())):
        match = frame_pattern.match(filename)
        if match:
            frame_num = int(match.group(1))
            frames.append(frame_num)

    # Identify consecutive bad frame sections
    bad_sections = []
    current_section = []

    for i in range(len(frames)):
        if i == 0 or frames[i] == frames[i - 1] + 1:
            current_section.append(frames[i])
        else:
            if len(current_section) >= 20:  # Change the condition to >= 8
                bad_sections.append((current_section[0], current_section[-1]))
            current_section = [frames[i]]

    # Add last section if it qualifies
    if len(current_section) >= 20:  # Change the condition to >= 8
        bad_sections.append((current_section[0], current_section[-1]))

    return {"bad_sections": bad_sections}


def generate_feedback(user_video_name, ref_video_name, low_score_segments, gemini_api_key):
    frames_folder = os.path.join("static", "uploads", "frames")
    bad_frames_data = detect_bad_frame_sections(frames_folder)
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    model = genai1.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)

    segments_info = [
        {"start_time": round(segment[0], 3), "end_time": round(segment[1], 3)} for segment in low_score_segments
    ]

    prompt = f"""You are analyzing movement differences between a reference video and an attempt to replicate it. Focus ONLY on the most obvious, major differences in movement.

    Context:
    - Reference Video: {ref_video_name}
    - Performer Video: {user_video_name}
    - Bad Frames: {json.dumps(bad_frames_data)}

    For each segment with low similarity:
    1. List only the 2-3 biggest differences in body position or movement
    2. Provide a brief, direct suggestion for improvement (50 - 100 words)

    Focus on:
    - Major differences in body positions
    - Obviously incorrect movements
    - Clear deviations in form

    Keep your analysis extremely simple and direct. Don't explain the biomechanics or impact - just point out what's different and how to fix it.
    example format, keep it the same as this format
    Avoid:
    - Minor details
    - Technical terminology
    - Long explanations
    - Multiple suggestions
    """

    client = genai.Client(api_key=gemini_api_key)

    files = [
        upload_to_gemini(user_video_name, mime_type="video/mp4"),
        upload_to_gemini(ref_video_name, mime_type="video/mp4"),
    ]
    wait_for_files_active(files)

    response = model.generate_content([
        prompt,
        files[0],
        files[1]
    ])
    print(response.text)
    return response.text


import re
import json


def process(videopath_1, videopath_2):
    kpts_vid1 = run(video_path=videopath_1)
    kpts_vid2 = run(video_path=videopath_2)
    aligned_frames1, aligned_frames2, path = align_videos(kpts_vid1, kpts_vid2)
    video_1_processed = similarity_preprocesser(aligned_frames1)
    video_2_processed = similarity_preprocesser(aligned_frames2)
    frame_scores = np.diag(cosine_similarity(video_1_processed, video_2_processed))
    scores = normalize_values(frame_scores / np.max(frame_scores))
    new_save_video(videopath_1, videopath_2, scores, path)
    #save_overlay_video(videopath_1, videopath_2, scores, path)
    low_score_segments = find_low_score_segments(scores)
    #parse_feedback(chichi)
    #save_video_segments(videopath_1, low_score_segments, path)
    #save_video_segments(videopath_2, low_score_segments, path)