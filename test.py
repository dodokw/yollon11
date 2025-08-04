import os
import cv2
import math
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp
import onnxruntime as ort

from tools.feature_extracter import FeatureExtractor
from tools.feature_summary import compute_demo_feature_df
from tools.utils.gaze import gaze_estimator
from tools.utils.attention import Attention, global_summarizer

# 경로 로드
data_path = 0

# 미디어파이프 모델 로드
face_detection_module = mp.solutions.face_detection
face_detect = face_detection_module.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
pose = mp.solutions.pose.Pose(static_image_mode=True)

# 기존 모델 로드
attention = Attention()

# feature 추출용 extractor 로드
feature_extractor = FeatureExtractor()

# user, label을 컬럼으로 둔 상태로 concat해도 되지만 일단 어느 정도 EDA 후에 그렇게 하자(각각 그룹바이해서 전처리하기 귀찮기 때문 ex. groupby.diff)
def compute_distance_change_from_index(df, x_col='x', y_col='y'):
    df = df.copy()
    df[x_col] = df[x_col].ffill().bfill().fillna(0)
    df[y_col] = df[y_col].ffill().bfill().fillna(0)
    df['delta_x'] = df[x_col].diff()
    df['delta_y'] = df[y_col].diff()
    return np.sqrt(df['delta_x']**2 + df['delta_y']**2)

# 랜드마크, 포즈 및 헤드포즈 저장용 얼굴 크롭 이미지를 모아놓는 함수
def detect_landmark_n_pose(cnt, image_draw, frame_rgb, frame,
                           data, headpose_data, original_data,
                           face_detect, face_mesh, pose, all_frame=300):
    baseline_frame = (all_frame // 10)

    # 랜드마크 추정
    temp_dict = {'frame': cnt}
    start_landmark = time.time()
    face_detect_result = face_detect.process(frame_rgb)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image_draw,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            h, w, _ = frame_rgb.shape
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_eye_center = np.mean(landmarks[left_eye_idx], axis=0)
            right_eye_center = np.mean(landmarks[right_eye_idx], axis=0)
            face_center = np.mean(landmarks, axis=0)
            left_ear = feature_extractor.calculate_ear(landmarks[left_eye_idx])
            right_ear = feature_extractor.calculate_ear(landmarks[right_eye_idx])
            left_eye_width = np.linalg.norm(landmarks[33] - landmarks[133])
            right_eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
            avg_ear = (left_ear + right_ear) / 2.0
            left_iris_idx = [468, 469, 470, 471]
            right_iris_idx = [473, 474, 475, 476]
            left_iris_center = np.mean(landmarks[left_iris_idx], axis=0)
            right_iris_center = np.mean(landmarks[right_iris_idx], axis=0)
            gaze_left = feature_extractor.estimate_gaze(left_iris_center, landmarks[33], landmarks[133])
            gaze_right = feature_extractor.estimate_gaze(right_iris_center, landmarks[362], landmarks[263])
            mar = feature_extractor.calculate_mar(landmarks)

            # bbox 기반 얼굴 영역 잘라서 head pose 추정
            if face_detect_result.detections:
                bbox = face_detect_result.detections[0].location_data.relative_bounding_box
                x_min = int(bbox.xmin * frame_rgb.shape[1])
                y_min = int(bbox.ymin * frame_rgb.shape[0])
                width = int(bbox.width * frame_rgb.shape[1])
                height = int(bbox.height * frame_rgb.shape[0])
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_min + width, frame_rgb.shape[1])
                y_max = min(y_min + height, frame_rgb.shape[0])
                # bbox 시각화
                cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                face_crop = frame[y_min:y_max, x_min:x_max]
                headpose_data.append({'frame': cnt, 'face_crop': face_crop})

            temp_dict.update({
                'ear_left': left_ear,
                'ear_right': right_ear,
                'ear_avg': avg_ear,
                'gaze_left': gaze_left,
                'gaze_right': gaze_right,
                'mar': mar,
                'left_eye_x': left_eye_center[0],
                'left_eye_y': left_eye_center[1],
                'right_eye_x': right_eye_center[0],
                'right_eye_y': right_eye_center[1],
                'left_iris_x': left_iris_center[0],
                'left_iris_y': left_iris_center[1],
                'right_iris_x': right_iris_center[0],
                'right_iris_y': right_iris_center[1],
                'face_center_x': face_center[0],
                'face_center_y': face_center[1],
                'left_eye_width': left_eye_width,
                'right_eye_width': right_eye_width,
            })
            if cnt % baseline_frame == 0:
                gaze: dict = gaze_estimator(
                    face_landmarks=face_landmarks,
                    img_w=w,
                    img_h=h,
                )
                original_temp_dict = {
                    'face_detect': face_detect_result.detections[0].score[0] if face_detect_result.detections else 0,
                    'gaze_theta': gaze["gaze_theta_left"],
                    'gaze_phi': gaze["gaze_phi_left"],
                    'gaze_ear': gaze["gaze_ear_left"],
                    'iris_radius': gaze["iris_radius_left"],
                }
    else:
        temp_dict.update({
            'ear_left': np.NaN,
            'ear_right': np.NaN,
            'ear_avg': np.NaN,
            'gaze_left': np.NaN,
            'gaze_right': np.NaN,
            'mar': np.NaN,
            'left_eye_x': np.NaN,
            'left_eye_y': np.NaN,
            'right_eye_x': np.NaN,
            'right_eye_y': np.NaN,
            'left_iris_x': np.NaN,
            'left_iris_y': np.NaN,
            'right_iris_x': np.NaN,
            'right_iris_y': np.NaN,
            'face_center_x': np.NaN,
            'face_center_y': np.NaN,
            'left_eye_width': np.NaN,
            'right_eye_width': np.NaN,
            'left_eye_width': np.NaN,
            'right_eye_width': np.NaN,
        })
        if cnt % baseline_frame == 0:
            original_temp_dict = {
                'face_detect': 0,
                'gaze_theta': 0,
                'gaze_phi': 0,
                'gaze_ear': 0,
                'iris_radius': 0,
            }
    # print("Landmark & headpose time:", time.time() - start_landmark)
    landmark_estimation_time = time.time() - start_landmark
    cv2.putText(image_draw, f'Landmark: {landmark_estimation_time:.3f}s', (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 포즈 추정
    start_pose = time.time()
    pose_results = pose.process(frame_rgb)
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image=image_draw,
            landmark_list=pose_results.pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        keypoints = pose_results.pose_landmarks.landmark
        # 포즈 랜드마크 bbox 시각화
        keypoints_xy = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in keypoints if
                        lm.visibility > 0.5]
        if keypoints_xy:
            x_coords, y_coords = zip(*keypoints_xy)
            x_min_pose, y_min_pose = min(x_coords), min(y_coords)
            x_max_pose, y_max_pose = max(x_coords), max(y_coords)
            cv2.rectangle(image_draw, (x_min_pose, y_min_pose), (x_max_pose, y_max_pose), (255, 0, 0), 2)
        for i, lm in enumerate(keypoints):
            temp_dict[f'pose_{i}_x'] = lm.x
            temp_dict[f'pose_{i}_y'] = lm.y
            temp_dict[f'pose_{i}_z'] = lm.z
            temp_dict[f'pose_{i}_visibility'] = lm.visibility
    else:
        for i in range(33):
            temp_dict[f'pose_{i}_x'] = None
            temp_dict[f'pose_{i}_y'] = None
            temp_dict[f'pose_{i}_z'] = None
            temp_dict[f'pose_{i}_visibility'] = None
    # print("Pose estimation time:", time.time() - start_pose)
    pose_estimation_time = time.time() - start_pose
    cv2.putText(image_draw, f'Pose: {pose_estimation_time:.3f}s', (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    data.append(temp_dict)

    # 오리지날 데이터는 초당 1프레임
    if cnt % baseline_frame == 0:
        original_data.append(original_temp_dict)

    return data, headpose_data, original_data, image_draw

# 랜드마크, 포즈 및 헤드포즈 저장용 얼굴 크롭 이미지를 사용하여 모델 입력값으로 변환
def preprocessing_for_modeing(data, headpose_data, image_draw, all_frame=300):
    df = pd.DataFrame(data)
    df['left_iris_dist'] = compute_distance_change_from_index(df, x_col='left_iris_x', y_col='left_iris_y')
    df['right_iris_dist'] = compute_distance_change_from_index(df, x_col='right_iris_x', y_col='right_iris_y')
    df['iris_dist'] = df['left_iris_dist'] + df['right_iris_dist'] / 2
    df['face_dist'] = compute_distance_change_from_index(df, x_col='face_center_x', y_col='face_center_y')

    df['left_gaze_vec_x'] = df['left_iris_x'] - df['left_eye_x']
    df['left_gaze_vec_y'] = df['left_iris_y'] - df['left_eye_y']
    df['right_gaze_vec_x'] = df['right_iris_x'] - df['right_eye_x']
    df['right_gaze_vec_y'] = df['right_iris_y'] - df['right_eye_y']
    df['left_gaze_dist_diff'] = compute_distance_change_from_index(df, x_col='left_gaze_vec_x',
                                                                   y_col='left_gaze_vec_y')
    df['right_gaze_dist_diff'] = compute_distance_change_from_index(df, x_col='right_gaze_vec_x',
                                                                    y_col='right_gaze_vec_y')
    df['gaze_dist_diff'] = df['left_gaze_dist_diff'] + df['right_gaze_dist_diff'] / 2
    df.drop([
        'left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y',
        'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
        'left_gaze_dist_diff', 'right_gaze_dist_diff',
        'face_center_x', 'face_center_y',
    ], axis=1, inplace=True)

    # 헤드포즈 모델링
    headpose_df = []
    for _, row in pd.DataFrame(headpose_data).iterrows():
        frame = row['frame']
        face_crop = row['face_crop']
        yaw, pitch, roll = feature_extractor.estimate_head_pose_onnx(face_crop)
        headpose_df.append([frame, yaw, pitch, roll])
    headpose_df = pd.DataFrame(headpose_df, columns=['frame', 'yaw', 'pitch', 'roll'])
    df = df.merge(headpose_df, on='frame', how='left')
    df[['yaw', 'pitch', 'roll']] = df[['yaw', 'pitch', 'roll']].fillna(np.nan)

    headpose_df_for_base_model = headpose_df[headpose_df['frame'] % (all_frame/10) == 0]
    headpose_df_for_base_model = headpose_df_for_base_model.drop(columns=['frame'])
    headpose_df_for_base_model = headpose_df_for_base_model.reset_index(drop=True)

    summary_df = compute_demo_feature_df(df=df,
                                         columns=['ear_avg', 'mar',
                                                  'iris_dist', 'face_dist',
                                                  'yaw', 'pitch', 'roll',
                                                  'left_gaze_vec_x', 'right_gaze_vec_x',
                                                  'gaze_dist_diff'
                                                  ],
                                         min_duration=math.ceil(all_frame * 0.05),
                                         )

    return summary_df, headpose_df_for_base_model, image_draw

# 랜드마크, 포즈 및 헤드포즈를 루프 내에서 한번에 추정하는 함수(시간 체크 필요)
def detect_landmark_n_pose_hpose(cnt, image_draw, frame_rgb, frame,
                           data, headpose_data, original_data,
                           face_detect, face_mesh, pose, all_frame=300):
    baseline_frame = (all_frame // 10)

    # 랜드마크 추정
    temp_dict = {'frame': cnt}
    start_landmark = time.time()
    face_detect_result = face_detect.process(frame_rgb)
    results = face_mesh.process(frame_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image_draw,
                landmark_list=face_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            h, w, _ = frame_rgb.shape
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_eye_center = np.mean(landmarks[left_eye_idx], axis=0)
            right_eye_center = np.mean(landmarks[right_eye_idx], axis=0)
            face_center = np.mean(landmarks, axis=0)
            left_ear = feature_extractor.calculate_ear(landmarks[left_eye_idx])
            right_ear = feature_extractor.calculate_ear(landmarks[right_eye_idx])
            left_eye_width = np.linalg.norm(landmarks[33] - landmarks[133])
            right_eye_width = np.linalg.norm(landmarks[362] - landmarks[263])
            avg_ear = (left_ear + right_ear) / 2.0
            left_iris_idx = [468, 469, 470, 471]
            right_iris_idx = [473, 474, 475, 476]
            left_iris_center = np.mean(landmarks[left_iris_idx], axis=0)
            right_iris_center = np.mean(landmarks[right_iris_idx], axis=0)
            gaze_left = feature_extractor.estimate_gaze(left_iris_center, landmarks[33], landmarks[133])
            gaze_right = feature_extractor.estimate_gaze(right_iris_center, landmarks[362], landmarks[263])
            mar = feature_extractor.calculate_mar(landmarks)

            # bbox 기반 얼굴 영역 잘라서 head pose 추정
            if face_detect_result.detections:
                bbox = face_detect_result.detections[0].location_data.relative_bounding_box
                x_min = int(bbox.xmin * frame_rgb.shape[1])
                y_min = int(bbox.ymin * frame_rgb.shape[0])
                width = int(bbox.width * frame_rgb.shape[1])
                height = int(bbox.height * frame_rgb.shape[0])
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(x_min + width, frame_rgb.shape[1])
                y_max = min(y_min + height, frame_rgb.shape[0])
                # bbox 시각화
                cv2.rectangle(image_draw, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                face_crop = frame[y_min:y_max, x_min:x_max]
                yaw, pitch, roll = feature_extractor.estimate_head_pose_onnx(face_crop)
            else:
                yaw = np.nan
                pitch = np.nan
                roll = np.nan

            temp_dict.update({
                'ear_left': left_ear,
                'ear_right': right_ear,
                'ear_avg': avg_ear,
                'gaze_left': gaze_left,
                'gaze_right': gaze_right,
                'mar': mar,
                'left_eye_x': left_eye_center[0],
                'left_eye_y': left_eye_center[1],
                'right_eye_x': right_eye_center[0],
                'right_eye_y': right_eye_center[1],
                'left_iris_x': left_iris_center[0],
                'left_iris_y': left_iris_center[1],
                'right_iris_x': right_iris_center[0],
                'right_iris_y': right_iris_center[1],
                'face_center_x': face_center[0],
                'face_center_y': face_center[1],
                'left_eye_width': left_eye_width,
                'right_eye_width': right_eye_width,
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
            })
            if cnt % baseline_frame == 0:
                gaze: dict = gaze_estimator(
                    face_landmarks=face_landmarks,
                    img_w=w,
                    img_h=h,
                )
                original_temp_dict = {
                    'face_detect': face_detect_result.detections[0].score[0] if face_detect_result.detections else 0,
                    'yaw': yaw,
                    'pitch': pitch,
                    'roll': roll,
                    'gaze_theta': gaze["gaze_theta_left"],
                    'gaze_phi': gaze["gaze_phi_left"],
                    'gaze_ear': gaze["gaze_ear_left"],
                    'iris_radius': gaze["iris_radius_left"],
                }
    else:
        temp_dict.update({
            'ear_left': np.nan,
            'ear_right': np.nan,
            'ear_avg': np.nan,
            'gaze_left': np.nan,
            'gaze_right': np.nan,
            'mar': np.nan,
            'left_eye_x': np.nan,
            'left_eye_y': np.nan,
            'right_eye_x': np.nan,
            'right_eye_y': np.nan,
            'left_iris_x': np.nan,
            'left_iris_y': np.nan,
            'right_iris_x': np.nan,
            'right_iris_y': np.nan,
            'face_center_x': np.nan,
            'face_center_y': np.nan,
            'left_eye_width': np.nan,
            'right_eye_width': np.nan,
            'left_eye_width': np.nan,
            'right_eye_width': np.nan,
            'yaw': np.nan,
            'pitch': np.nan,
            'roll': np.nan,
        })
        if cnt % baseline_frame == 0:
            original_temp_dict = {
                'face_detect': 0,
                'yaw': 0,
                'pitch': 0,
                'roll': 0,
                'gaze_theta': 0,
                'gaze_phi': 0,
                'gaze_ear': 0,
                'iris_radius': 0,
            }
    # print("Landmark & headpose time:", time.time() - start_landmark)
    landmark_estimation_time = time.time() - start_landmark
    cv2.putText(image_draw, f'Landmark: {landmark_estimation_time:.3f}s', (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 포즈 추정
    start_pose = time.time()
    pose_results = pose.process(frame_rgb)
    if pose_results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image=image_draw,
            landmark_list=pose_results.pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        keypoints = pose_results.pose_landmarks.landmark
        # 포즈 랜드마크 bbox 시각화
        keypoints_xy = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in keypoints if
                        lm.visibility > 0.5]
        if keypoints_xy:
            x_coords, y_coords = zip(*keypoints_xy)
            x_min_pose, y_min_pose = min(x_coords), min(y_coords)
            x_max_pose, y_max_pose = max(x_coords), max(y_coords)
            cv2.rectangle(image_draw, (x_min_pose, y_min_pose), (x_max_pose, y_max_pose), (255, 0, 0), 2)
        for i, lm in enumerate(keypoints):
            temp_dict[f'pose_{i}_x'] = lm.x
            temp_dict[f'pose_{i}_y'] = lm.y
            temp_dict[f'pose_{i}_z'] = lm.z
            temp_dict[f'pose_{i}_visibility'] = lm.visibility
    else:
        for i in range(33):
            temp_dict[f'pose_{i}_x'] = None
            temp_dict[f'pose_{i}_y'] = None
            temp_dict[f'pose_{i}_z'] = None
            temp_dict[f'pose_{i}_visibility'] = None
    # print("Pose estimation time:", time.time() - start_pose)
    pose_estimation_time = time.time() - start_pose
    cv2.putText(image_draw, f'Pose: {pose_estimation_time:.3f}s', (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    data.append(temp_dict)

    # 오리지날 데이터는 초당 1프레임
    if cnt % baseline_frame == 0:
        original_data.append(original_temp_dict)

    return data, original_data, image_draw

# 헤드포즈 연산 없는 전처리
def preprocessing_for_modeing_not_use_fsa(data, image_draw, all_frame=300):

    df = pd.DataFrame(data)
    df['left_iris_dist'] = compute_distance_change_from_index(df, x_col='left_iris_x', y_col='left_iris_y')
    df['right_iris_dist'] = compute_distance_change_from_index(df, x_col='right_iris_x', y_col='right_iris_y')
    df['iris_dist'] = df['left_iris_dist'] + df['right_iris_dist'] / 2
    df['face_dist'] = compute_distance_change_from_index(df, x_col='face_center_x', y_col='face_center_y')

    df['left_gaze_vec_x'] = df['left_iris_x'] - df['left_eye_x']
    df['left_gaze_vec_y'] = df['left_iris_y'] - df['left_eye_y']
    df['right_gaze_vec_x'] = df['right_iris_x'] - df['right_eye_x']
    df['right_gaze_vec_y'] = df['right_iris_y'] - df['right_eye_y']
    df['left_gaze_dist_diff'] = compute_distance_change_from_index(df, x_col='left_gaze_vec_x',
                                                                   y_col='left_gaze_vec_y')
    df['right_gaze_dist_diff'] = compute_distance_change_from_index(df, x_col='right_gaze_vec_x',
                                                                    y_col='right_gaze_vec_y')
    df['gaze_dist_diff'] = df['left_gaze_dist_diff'] + df['right_gaze_dist_diff'] / 2
    df.drop([
        'left_iris_x', 'left_iris_y', 'right_iris_x', 'right_iris_y',
        'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
        'left_gaze_dist_diff', 'right_gaze_dist_diff',
        'face_center_x', 'face_center_y',
    ], axis=1, inplace=True)

    summary_df = compute_demo_feature_df(df=df,
                                         columns=['ear_avg', 'mar',
                                                  'iris_dist', 'face_dist',
                                                  'yaw', 'pitch', 'roll',
                                                  'left_gaze_vec_x', 'right_gaze_vec_x',
                                                  'gaze_dist_diff'
                                                  ],
                                         min_duration=math.ceil(all_frame * 0.05),
                                         )

    return summary_df, image_draw

# 추론
def inference(summary_df, selected_features, labels_dict, ort_session):
    input_features = summary_df[selected_features].astype(np.float32)
    input_features = input_features.fillna(0).to_numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: input_features}
    ort_outputs = ort_session.run(None, ort_inputs)

    def softmax(x):
        e_x = np.exp(x - np.max(x))  # 안정성을 위한 max값 보정
        return e_x / e_x.sum()

    probs = softmax(ort_outputs[1])[0].round(2)
    result = labels_dict.get(ort_outputs[0][0])

    return result, probs

# 메인 함수
def main(video_path=0, all_frame=300):

    # 합친 레이블
    labels_dict = {
        0: 'Study',
        1: 'Distraction',
        2: 'Spacing out',
        3: 'Sleep'
    }

    # xgboost 로드
    onnx_model_path = f"onnx_models/xgboost_{all_frame}frame.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)

    # 초기화 변수들
    data = []
    original_data = []
    headpose_data = []
    cnt = 0
    result = None
    atntn_score = None
    probs = None

    if str(video_path).isdigit():
        video_path = int(video_path)
    else:
        all_frame = 300
    cap = cv2.VideoCapture(video_path)

    loop_start = time.time()
    while cap.isOpened():
        temp_loop_start = time.time()
        success, frame = cap.read()
        cnt += 1
        if not success:
            break

        # extract features
        image_draw = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data, headpose_data, original_data, image_draw = detect_landmark_n_pose(
            cnt, image_draw, frame_rgb, frame,
            data, headpose_data, original_data,
            face_detect, face_mesh, pose,
            all_frame=all_frame)

        temp_loop_end = time.time() - temp_loop_start
        cv2.putText(image_draw, f'Loop 1: {temp_loop_end:.3f}s', (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        if cnt == all_frame:
            # 데이터 전처리
            all_capture_time = time.time() - loop_start
            start_pre = time.time()

            summary_df, headpose_df_for_base_model, image_draw = preprocessing_for_modeing(data, headpose_data, image_draw, all_frame)
            cv2.putText(image_draw, f'Preproc: {time.time() - start_pre:.3f}s', (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

            # 모델 적용
            # xgboost 인퍼런스
            start_model = time.time()
            selected_features = [
                'face_detect_count',
                'ear_avg_minmax', 'ear_avg_std', 'ear_avg_mean',  # std, minmax, iqr 등도 비슷
                'ear_avg_blink_count',  # 멍때림과 영상시청 차이(차이가 크지는 않음)
                'mar_minmax', 'mar_std',  # std minmax, max_diff 등도 비슷
                'mar_yawn_count',  # 이 지표에서 확인은 어렵지만 하품=딴짓 카운트가 꽤 성능이 좋음
                'mar_mean_abs_diff',
                'iris_dist_minmax', 'iris_dist_std',
                'face_dist_minmax', 'face_dist_std',
                'yaw_std', 'yaw_minmax', 'yaw_headturns_sign',
                'pitch_std', 'pitch_minmax', 'pitch_headturns_diff',
                'roll_std', 'roll_minmax', 'roll_headturns_diff', 'roll_mean',
                'left_gaze_vec_x_std', 'left_gaze_vec_x_minmax',
                'gaze_dist_diff_std', 'gaze_dist_diff_minmax',
            ]

            print(summary_df.loc[:,selected_features].to_markdown())

            result, probs = inference(summary_df, selected_features, labels_dict, ort_session)
            cv2.putText(image_draw, f'Model: {time.time() - start_model:.3f}s', (10, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            # 오리지날 모델 적용
            combined_df = pd.concat(
                [
                    pd.DataFrame(original_data, columns=['face_detect', 'gaze_theta', 'gaze_phi', 'gaze_ear', 'iris_radius']).reset_index(drop=True),
                    headpose_df_for_base_model.reset_index(drop=True)
                ],
                axis=1
            )
            combined_df = combined_df[['face_detect', 'yaw', 'pitch', 'roll', 'gaze_theta', 'gaze_phi', 'gaze_ear', 'iris_radius']]
            global_features = global_summarizer(pd.DataFrame(combined_df))
            atntn_score: float = attention.atntn_scoring(global_features)
            atntn_score = round(atntn_score,2)
            # 결과 프린트
            print(
                f"모델 결과: {str(result):<11} | "
                f"베이스라인 집중력: {atntn_score:>5.2f}점 | "
                f"캡쳐: {all_capture_time:>6.3f}s | "
                f"전처리(+HPose): {time.time() - start_pre:>6.3f}s | "
                f"모델: {time.time() - start_model:>6.3f}s | "
                f"전체: {time.time() - loop_start:>6.3f}s"
            )
            # 초기화
            cnt = 0
            data = []
            original_data = []
            headpose_data = []
            loop_start = time.time()

            # image_draw에 cv.puttext로 좌상단에 행동 분류 결과(result) 프린트 강도 2
            cv2.putText(image_draw, f'Action: {result}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Probs: {probs}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Base Model Score: {atntn_score}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 250), 2)
            cv2.imshow('Pose Detection Comparison', image_draw)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            # image_draw에 cv.puttext로 좌상단에 행동 분류 결과(result) 프린트 강도 2
            cv2.putText(image_draw, f'Action: {result}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Probs: {probs}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Base Model Score: {atntn_score}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 250), 2)
            cv2.imshow('Pose Detection Comparison', image_draw)

            min_interval = 1/(all_frame//10) # FPS 유지를 위함
            if temp_loop_end < min_interval:
                wait_time_ms = int((min_interval - temp_loop_end) * 1000)
                cv2.waitKey(wait_time_ms)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# 만약에 headpose + loop가 충분히 10초 내에 가능할 경우 이 함수 사용
def main_fast(video_path=0, all_frame=300):
    # 합친 레이블
    labels_dict = {
        0: 'Study',
        1: 'Distraction',
        2: 'Spacing out',
        3: 'Sleep'
    }

    # xgboost 로드
    onnx_model_path = f"onnx_models/xgboost_{all_frame}frame.onnx"
    ort_session = ort.InferenceSession(onnx_model_path)

    # 초기화 변수들
    data = []
    original_data = []
    headpose_data = []
    cnt = 0
    result = None
    atntn_score = None
    probs = None

    if str(video_path).isdigit():
        video_path = int(video_path)
    else:
        all_frame = 300
    cap = cv2.VideoCapture(video_path)

    loop_start = time.time()
    while cap.isOpened():
        temp_loop_start = time.time()
        success, frame = cap.read()
        cnt += 1
        if not success:
            break

        # extract features
        image_draw = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data, original_data, image_draw = detect_landmark_n_pose_hpose(
            cnt, image_draw, frame_rgb, frame,
            data, headpose_data, original_data,
            face_detect, face_mesh, pose,
            all_frame=all_frame
        )

        temp_loop_end = time.time() - temp_loop_start
        cv2.putText(image_draw, f'Loop 1: {temp_loop_end:.3f}s', (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        if cnt == all_frame:
            # 데이터 전처리
            all_capture_time = time.time() - loop_start
            start_pre = time.time()

            summary_df, image_draw = preprocessing_for_modeing_not_use_fsa(data, image_draw, all_frame)
            cv2.putText(image_draw, f'Preproc: {time.time() - start_pre:.3f}s', (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 0), 2)

            # 모델 적용
            # xgboost 인퍼런스
            start_model = time.time()
            selected_features = [
                'face_detect_count',
                'ear_avg_minmax', 'ear_avg_std', 'ear_avg_mean',  # std, minmax, iqr 등도 비슷
                'ear_avg_blink_count',  # 멍때림과 영상시청 차이(차이가 크지는 않음)
                'mar_minmax', 'mar_std',  # std minmax, max_diff 등도 비슷
                'mar_yawn_count',  # 이 지표에서 확인은 어렵지만 하품=딴짓 카운트가 꽤 성능이 좋음
                'mar_mean_abs_diff',
                'iris_dist_minmax', 'iris_dist_std',
                'face_dist_minmax', 'face_dist_std',
                'yaw_std', 'yaw_minmax', 'yaw_headturns_sign',
                'pitch_std', 'pitch_minmax', 'pitch_headturns_diff',
                'roll_std', 'roll_minmax', 'roll_headturns_diff', 'roll_mean',
                'left_gaze_vec_x_std', 'left_gaze_vec_x_minmax',
                'gaze_dist_diff_std', 'gaze_dist_diff_minmax',
            ]

            print(summary_df.loc[:, selected_features].to_markdown())

            result, probs = inference(summary_df, selected_features, labels_dict, ort_session)
            cv2.putText(image_draw, f'Model: {time.time() - start_model:.3f}s', (10, 310),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

            # 오리지날 모델 적용
            global_features = global_summarizer(pd.DataFrame(original_data))
            atntn_score: float = attention.atntn_scoring(global_features)
            atntn_score = round(atntn_score, 2)
            # 결과 프린트
            print(
                f"모델 결과: {str(result):<11} | "
                f"베이스라인 집중력: {atntn_score:>5.2f}점 | "
                f"캡쳐: {all_capture_time:>6.3f}s | "
                f"전처리: {time.time() - start_pre:>6.3f}s | "
                f"모델: {time.time() - start_model:>6.3f}s | "
                f"전체: {time.time() - loop_start:>6.3f}s"
            )
            # 초기화
            cnt = 0
            data = []
            original_data = []
            headpose_data = []
            loop_start = time.time()

            # image_draw에 cv.puttext로 좌상단에 행동 분류 결과(result) 프린트 강도 2
            cv2.putText(image_draw, f'Action: {result}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Probs: {probs}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Base Model Score: {atntn_score}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 250), 2)
            cv2.imshow('Pose Detection Comparison', image_draw)
            if cv2.waitKey(5) & 0xFF == 27:
                break
        else:
            # image_draw에 cv.puttext로 좌상단에 행동 분류 결과(result) 프린트 강도 2
            cv2.putText(image_draw, f'Action: {result}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Probs: {probs}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image_draw, f'Base Model Score: {atntn_score}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 250), 2)
            cv2.imshow('Pose Detection Comparison', image_draw)

            min_interval = 1 / (all_frame // 10)  # FPS 유지를 위함
            if temp_loop_end < min_interval:
                wait_time_ms = int((min_interval - temp_loop_end) * 1000)
                cv2.waitKey(wait_time_ms)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    video_sources = {
        1: '/Volumes/backup/dkcnsengage_4/data/20250612/104100/104100_4_5.mp4',  # 소장님 공부 정자세 (TFLITE 0.95)
        2: '/Volumes/backup/dkcnsengage_4/data/20250609/135733/135733_4_5.mp4',  # 은지책임님 어두운 화면 (TFLITE 0.85)
        3: '/Volumes/backup/dkcnsengage_4/data/20250509/101354/101354_4_183.mp4',  # 나 머리카락때문에 잘 안보임 (TFLITE 0.80 & 페이스매쉬 실패값 있음)
        4: '/Volumes/backup/dkcnsengage_4/data/20250605/105407/105407_4_8.mp4',  # 동관님 완전 엎드린 자세 (TFLITE 0.50 & 페이스매쉬 전체 실패) #
        5: '/Volumes/backup/dkcnsengage_4/data/20250604/154349/154349_4_7.mp4',  # 봉석수석님 기울여서 엎드린 자세 (TFLITE 0.80 & 페이스매쉬 전체 실패) # sleep 83.5
        6: '/Volumes/backup/dkcnsengage/data/20250612/104100/104100_8_5.mp4',  # 소장님 공부 정자세 (TFLITE 0.90)  # space out 0.2 ---> 움직임이 적나?
        7: '/Volumes/backup/dkcnsengage/data/20250609/135733/135733_8_5.mp4',  # 은지책임님 어두운 화면 (TFLITE 0.80)  # study 0.3
        8: '/Volumes/backup/dkcnsengage/data/20250509/101354/101354_8_183.mp4',  # 나 머리카락때문에 잘 안보임 (TFLITE 0.80)  # study 0.6
        9: '/Volumes/backup/dkcnsengage/data/20250605/105407/105407_8_8.mp4',  # 동관님 완전 엎드린 자세 (TFLITE 0.50 & 페이스매쉬 대부분 실패)
        10: '/Volumes/backup/dkcnsengage/data/20250604/154349/154349_8_7.mp4',  # 봉석수석님 기울여서 엎드린 자세, 이건 근데 김우빈나옴..
        11: '/Volumes/backup/dkcnsengage_4/data/20250528/101538/101538_4_10.mp4', # 동규님 엎드린 자세 (TFLITE 0.50)
        12: '/Volumes/backup/dkcnsengage/data/20250528/101538/101538_8_10.mp4',  # 동규님 엎드린 자세 (TFLITE 0.50)
        13: '/Volumes/backup/dkcnsengage_4/data/20250612/143900/143900_4_10.mp4', # 종찬님 머리카락때문에 얼굴 가려짐 (TFLITE 0.80)
        14: '/Volumes/backup/dkcnsengage/data/20250509/101354/101354_7_183.mp4'
    }

    main_fast(
        video_path=0,
        all_frame=30
    ) # video_sources[11]