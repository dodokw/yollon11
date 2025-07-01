/**
 * App.tsx
 *
 * 이 파일은 애플리케이션의 메인 진입점입니다.
 * 카메라 설정, 모델 로딩, 그리고 포즈 오버레이 렌더링을 담당합니다.
 */
import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from 'react';
import { StyleSheet, Text, View, Dimensions, Platform } from 'react-native';
import {
  Camera,
  Frame,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Svg, Circle, Line } from 'react-native-svg';
import Animated, {
  useSharedValue,
  useAnimatedProps,
} from 'react-native-reanimated';
import { InferenceSession, Tensor } from 'onnxruntime-react-native';
import RNFS from 'react-native-fs';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS } from 'react-native-worklets-core';

// 화면 크기 가져오기
const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// ONNX 모델 관련 상수
const MODEL_INPUT_WIDTH = 640;
const MODEL_INPUT_HEIGHT = 640;

// 후처리 관련 상수
const CONFIDENCE_THRESHOLD = 0.5;
const KEYPOINT_CONFIDENCE_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.5;

// 타입 정의
interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}

interface Pose {
  box: [number, number, number, number];
  keypoints: Keypoint[];
  score: number;
}

// Reanimated와 SVG를 연결하기 위한 컴포넌트 생성
const AnimatedCircle = Animated.createAnimatedComponent(Circle);
const AnimatedLine = Animated.createAnimatedComponent(Line);

// COCO 키포인트 연결 순서 정의
const SKELETON = [
  [16, 14],
  [14, 12],
  [17, 15],
  [15, 13],
  [12, 13],
  [6, 12],
  [7, 13],
  [6, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [9, 11],
  [2, 3],
  [1, 2],
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 6],
  [5, 7],
].map(pair => [pair[0] - 1, pair[1] - 1]);

// --- 유틸리티 함수 ---

const nonMaxSuppression = (
  boxes: number[][],
  scores: number[],
  iouThreshold: number,
): number[] => {
  if (boxes.length === 0) return [];

  const indices = Array.from(Array(boxes.length).keys());
  indices.sort((a, b) => scores[b] - scores[a]);

  const keep = [];
  while (indices.length > 0) {
    const i = indices.shift()!;
    keep.push(i);
    for (let j = indices.length - 1; j >= 0; j--) {
      const k = indices[j];
      const box1 = boxes[i];
      const box2 = boxes[k];

      const xA = Math.max(box1[0], box2[0]);
      const yA = Math.max(box1[1], box2[1]);
      const xB = Math.min(box1[2], box2[2]);
      const yB = Math.min(box1[3], box2[3]);

      const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
      const box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
      const box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
      const iou = interArea / (box1Area + box2Area - interArea);

      if (iou > iouThreshold) {
        indices.splice(j, 1);
      }
    }
  }
  return keep;
};

// --- 타입 정의 (기존 코드와 동일) ---
interface Keypoint {
  x: number;
  y: number;
  confidence: number;
}

interface Pose {
  box: [number, number, number, number];
  keypoints: Keypoint[];
  score: number;
}

// --- 유틸리티 함수 ---

/**
 * 1차원 배열 데이터를 [열][행] 구조의 2차원 배열로 변환 (전치)
 * @param data 원본 Float32Array 데이터
 * @param rows 원본 텐서의 행 수 (numFeatures, 예: 56)
 * @param cols 원본 텐서의 열 수 (numDetections, 예: 8400)
 * @returns 전치된 2차원 배열 (예: 8400 x 56)
 */
const transpose = (
  data: Float32Array,
  rows: number,
  cols: number,
): number[][] => {
  const result: number[][] = Array.from({ length: cols }, () => []);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      // result[j]가 undefined가 아니도록 초기화 필요
      result[j][i] = data[i * cols + j];
    }
  }
  return result;
};

const processOutput = (output: Tensor): Pose[] => {
  // 반환 타입이 Pose[] (배열) 이어야 합니다.
  console.log('processOutput 실행 확인');
  const data = output.data as Float32Array;
  // 모델 출력 형태: [1, 56, 8400]
  // dims[0] = 1 (batch)
  // dims[1] = 56 (features)
  // dims[2] = 8400 (detections)
  const numFeatures = output.dims[1];
  const numDetections = output.dims[2];

  // 1. 텐서 전치: [56, 8400] -> [8400, 56]
  const transposedData = transpose(data, numFeatures, numDetections);

  const boxes: number[][] = [];
  const scores: number[] = [];
  const allKeypoints: Keypoint[][] = [];

  // 2. 8400개의 잠재적 탐지 후보를 순회
  for (const detection of transposedData) {
    // 3. 객체 신뢰도 점수 확인 (인덱스 4)
    const score = detection[4];
    if (score < CONFIDENCE_THRESHOLD) {
      continue;
    }

    // 바운딩 박스 정보 추출 (cx, cy, w, h)
    const cx = detection[0];
    const cy = detection[1];
    const w = detection[2];
    const h = detection[3];

    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    boxes.push([x1, y1, x2, y2]);
    scores.push(score);

    const keypoints: Keypoint[] = [];
    // 4. 17개의 키포인트 데이터 추출
    // 키포인트 데이터는 인덱스 5부터 시작합니다.
    for (let j = 0; j < 17; j++) {
      const kptOffset = 5 + j * 3;
      keypoints.push({
        x: detection[kptOffset],
        y: detection[kptOffset + 1],
        confidence: detection[kptOffset + 2],
      });
    }
    allKeypoints.push(keypoints);
  }

  // 5. Non-Max Suppression 실행
  const indices = nonMaxSuppression(boxes, scores, IOU_THRESHOLD);
  console.log('NMS 후 유효한 탐지 수:', indices.length);

  // 최종 포즈 데이터를 생성하여 반환
  return indices.map(idx => ({
    box: boxes[idx] as [number, number, number, number],
    score: scores[idx],
    keypoints: allKeypoints[idx],
  }));
};

// --- PoseOverlay 컴포넌트 및 하위 컴포넌트 ---

// 단일 뼈대를 렌더링하는 최적화된 컴포넌트
// React Hook 규칙을 준수하기 위해 별도 컴포넌트로 분리합니다.
const AnimatedBone = ({
  poses,
  poseIndex,
  bone,
  scaleX,
  scaleY,
  offsetX,
  offsetY,
}: {
  poses: Animated.SharedValue<Pose[]>;
  poseIndex: number;
  bone: number[];
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}) => {
  // useAnimatedProps는 컴포넌트의 최상위 레벨에서 호출되어야 합니다.
  const animatedProps = useAnimatedProps(() => {
    const pose = poses.value[poseIndex];
    // 포즈가 없거나, 키포인트 신뢰도가 낮으면 렌더링하지 않습니다.
    if (!pose) return { display: 'none' };

    const kp1 = pose.keypoints[bone[0]];
    const kp2 = pose.keypoints[bone[1]];
    if (
      !kp1 ||
      !kp2 ||
      kp1.confidence < KEYPOINT_CONFIDENCE_THRESHOLD ||
      kp2.confidence < KEYPOINT_CONFIDENCE_THRESHOLD
    ) {
      return { display: 'none' };
    }
    return {
      x1: kp1.x * scaleX + offsetX,
      y1: kp1.y * scaleY + offsetY,
      x2: kp2.x * scaleX + offsetX,
      y2: kp2.y * scaleY + offsetY,
      display: 'flex',
    };
  });

  return (
    <AnimatedLine animatedProps={animatedProps} stroke="lime" strokeWidth="2" />
  );
};

// 단일 키포인트를 렌더링하는 최적화된 컴포넌트
const AnimatedKeypoint = ({
  poses,
  poseIndex,
  keypointIndex,
  scaleX,
  scaleY,
  offsetX,
  offsetY,
}: {
  poses: Animated.SharedValue<Pose[]>;
  poseIndex: number;
  keypointIndex: number;
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}) => {
  // useAnimatedProps는 컴포넌트의 최상위 레벨에서 호출되어야 합니다.
  const animatedProps = useAnimatedProps(() => {
    const pose = poses.value[poseIndex];
    if (!pose) return { display: 'none' };

    const keypoint = pose.keypoints[keypointIndex];
    if (!keypoint || keypoint.confidence < KEYPOINT_CONFIDENCE_THRESHOLD) {
      return { display: 'none' };
    }
    return {
      cx: keypoint.x * scaleX + offsetX,
      cy: keypoint.y * scaleY + offsetY,
      r: 5,
      fill: 'red',
      display: 'flex',
    };
  });

  return <AnimatedCircle animatedProps={animatedProps} />;
};

// 단일 포즈(뼈대 + 키포인트)를 렌더링하는 컴포넌트
const SinglePose = ({
  poses,
  poseIndex,
  scaleX,
  scaleY,
  offsetX,
  offsetY,
}: {
  poses: Animated.SharedValue<Pose[]>;
  poseIndex: number;
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
}) => {
  return (
    <React.Fragment>
      {SKELETON.map((bone, boneIndex) => (
        <AnimatedBone
          key={`bone-${poseIndex}-${boneIndex}`}
          poses={poses}
          poseIndex={poseIndex}
          bone={bone}
          scaleX={scaleX}
          scaleY={scaleY}
          offsetX={offsetX}
          offsetY={offsetY}
        />
      ))}
      {Array.from({ length: 17 }).map((_, keypointIndex) => (
        <AnimatedKeypoint
          key={`kp-${poseIndex}-${keypointIndex}`}
          poses={poses}
          poseIndex={poseIndex}
          keypointIndex={keypointIndex}
          scaleX={scaleX}
          scaleY={scaleY}
          offsetX={offsetX}
          offsetY={offsetY}
        />
      ))}
    </React.Fragment>
  );
};

/**
 * PoseOverlay 컴포넌트
 */
const PoseOverlay = ({
  poses,
  frameWidth,
  frameHeight,
}: {
  poses: Animated.SharedValue<Pose[]>;
  frameWidth: number;
  frameHeight: number;
}) => {
  // 1. 화면과 프레임의 종횡비 계산
  const screenAspectRatio = screenWidth / screenHeight;
  const frameAspectRatio = frameWidth / frameHeight;

  // 2. 렌더링될 프리뷰의 크기와 오프셋 계산
  // react-native-vision-camera는 기본적으로 'cover' 방식으로 화면을 채웁니다.
  // 이로 인해 발생하는 레터박스/필러박스를 계산하여 좌표를 보정해야 합니다.
  let offsetX = 0;
  let offsetY = 0;
  let previewWidth = screenWidth;
  let previewHeight = screenHeight;

  if (frameAspectRatio > screenAspectRatio) {
    // 프레임이 화면보다 넓은 경우 (세로가 긴 화면), 프리뷰는 화면 높이에 맞춰지고 가로가 잘립니다.
    // 실제 보이는 영상의 너비를 계산합니다.
    previewWidth = screenHeight * frameAspectRatio;
    offsetX = (screenWidth - previewWidth) / 2;
  } else {
    // 프레임이 화면보다 좁거나 같은 경우 (가로가 긴 화면), 프리뷰는 화면 너비에 맞춰지고 세로가 잘립니다.
    // 실제 보이는 영상의 높이를 계산합니다.
    previewHeight = screenWidth / frameAspectRatio;
    offsetY = (screenHeight - previewHeight) / 2;
  }

  // 3. 모델 입력 크기(640x640)에서 실제 화면에 보이는 프리뷰 크기로의 스케일 팩터 계산
  const scaleX = previewWidth / MODEL_INPUT_WIDTH;
  const scaleY = previewHeight / MODEL_INPUT_HEIGHT;

  return (
    <Svg
      width={screenWidth}
      height={screenHeight}
      style={StyleSheet.absoluteFill}
    >
      {Array.from({ length: 10 }).map((_, poseIndex) => (
        <SinglePose
          key={`pose-${poseIndex}`}
          poses={poses}
          poseIndex={poseIndex}
          scaleX={scaleX}
          scaleY={scaleY}
          offsetX={offsetX}
          offsetY={offsetY}
        />
      ))}
    </Svg>
  );
};

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const { resize } = useResizePlugin();

  const format = useMemo(() => {
    if (!device?.formats) return undefined;

    // 1. 사용 가능한 모든 포맷을 가져옵니다.
    const allFormats = device.formats;

    // 2. 너무 높은 해상도(예: 4K)를 제외하고, 15 FPS 이상을 지원하는 포맷만 필터링합니다.
    // 1920x1080 (FHD) 이하의 해상도를 우선적으로 고려합니다.
    const filteredFormats = allFormats.filter(
      f => f.videoWidth <= 1920 && f.maxFps >= 15,
    );

    if (filteredFormats.length > 0) {
      // 3. 성능을 위해, 필터링된 포맷들을 해상도(너비 기준) 오름차순으로 정렬합니다.
      // 이렇게 하면 가장 낮은 해상도를 가진 포맷을 선택할 수 있습니다.
      filteredFormats.sort((a, b) => a.videoWidth - b.videoWidth);

      // 4. 가장 낮은 해상도를 가진 포맷을 선택합니다.
      const lowestResFormat = filteredFormats[0];
      console.log(
        'App: 가장 낮은 해상도 포맷 선택:',
        lowestResFormat.videoWidth,
        'x',
        lowestResFormat.videoHeight,
        '@',
        lowestResFormat.maxFps,
        'fps',
      );
      return lowestResFormat;
    } else {
      console.warn(
        'App: 1920x1080 & 15fps 이상을 만족하는 포맷을 찾지 못했습니다. 사용 가능한 첫 번째 포맷으로 대체합니다.',
      );
      // 적합한 포맷이 없으면 첫 번째 포맷을 사용 (fallback)
      return device.formats.length > 0 ? device.formats[0] : undefined;
    }
  }, [device?.formats]);

  const frameWidth = format?.videoWidth ?? 0;
  const frameHeight = format?.videoHeight ?? 0;

  const [session, setSession] = useState<InferenceSession | null>(null);
  const detectedPoses = useSharedValue<Pose[]>([]);
  // Worklet과 JS 스레드 간의 상태를 안전하게 공유하기 위해 useSharedValue를 사용합니다.
  // useRef는 스레드 간 동기화를 보장하지 않아 Worklet이 오래된 값을 볼 수 있습니다.
  const isProcessing = useSharedValue(false);
  const lastInferenceTime = useRef(0);

  useEffect(() => {
    (async () => {
      await requestPermission();
      try {
        let modelPath: string;
        const modelFilename = 'yolo11n-pose.onnx';

        if (Platform.OS === 'android') {
          const modelDestPath = `${RNFS.DocumentDirectoryPath}/${modelFilename}`;

          if (!(await RNFS.existsAssets(modelFilename))) {
            throw new Error(
              `모델 파일(${modelFilename})이 Android assets 폴더에 없습니다.`,
            );
          }

          if (!(await RNFS.exists(modelDestPath))) {
            console.log('모델을 assets에서 내부 저장소로 복사합니다...');
            await RNFS.copyFileAssets(modelFilename, modelDestPath);
            console.log('모델 복사 완료.');
          }
          modelPath = modelDestPath;
        } else {
          modelPath = `${RNFS.MainBundlePath}/${modelFilename}`;
        }

        console.log(`모델 로딩 경로: ${modelPath}`);
        // 하드웨어 가속 옵션 추가(추가 테스트)
        // const sessionOptions: InferenceSession.SessionOptions = {
        //   executionProviders: ['CUDAExecutionProvider', 'CPUExecutionProvider'],
        // };
        const newSession = await InferenceSession.create(modelPath);
        setSession(newSession);
        console.log('ONNX 세션이 성공적으로 로드되었습니다.');
      } catch (e) {
        console.error('모델 로드 실패:', e);
      }
    })();
  }, [requestPermission]);

  // 추론 함수 - useRunOnJS를 사용하여 Worklet에서 호출 가능하도록 만듭니다.
  const runInference = useRunOnJS(
    async (tensorDataArray: number[]) => {
      console.log('runInference::');
      if (!session) {
        console.log('세션이 없습니다.');
        isProcessing.value = false;
        return;
      }
      console.log('세션이 있습니다.', session);

      try {
        // 일반 배열을 Float32Array로 변환
        const tensorData = new Float32Array(tensorDataArray);

        // Tensor 생성 시 데이터 타입을 명시적으로 지정하는 것이 안전합니다.
        const tensor = new Tensor('float32', tensorData, [
          1,
          3,
          MODEL_INPUT_HEIGHT,
          MODEL_INPUT_WIDTH,
        ]);
        console.log('tensor::::생성');

        const feed = { images: tensor };
        const results = await session.run(feed);
        console.log('results:::cameout', Object.keys(results));
        //keyvalue check후 수정************
        // onnxruntime-react-native의 run 메서드는 출력 이름을 키로 갖는 객체를 반환합니다.
        // 로그에서 확인된 키 'output0'을 사용해야 합니다.
        const outputTensor = results['output0'];
        if (outputTensor) {
          console.log('processOutput:::실행');
          const poses = processOutput(outputTensor);
          // console.log('poses:::::', poses);
          console.log('poses::::::');
          detectedPoses.value = poses;
        } else {
          console.log(
            '모델 출력에서 "output0"을 찾을 수 없습니다. 사용 가능한 키:',
            Object.keys(results),
          );
        }
      } catch (e) {
        console.error('추론 오류:', e);
      } finally {
        isProcessing.value = false;
        console.log('runInference fin');
      }
      console.log('*********fin runInference*********');
    },
    [session],
  );

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet';

      if (!session) {
        //   // isProcessing.value가 true이면, 이전 프레임이 아직 JS 스레드에서 처리 중이라는 의미입니다.
        //   // console.log('여기에 머무는동안 로그가 얼마나 찍히는지 확인');
        //   // console.log('session:::::', session);
        return;
      }

      // 2초에 한 번만 추론을 실행하도록 조절 (Throttling)
      const now = Date.now();
      if (now - lastInferenceTime.current < 700) {
        return;
      }
      // 실제 처리를 시작할 때 마지막 실행 시간을 기록합니다.
      lastInferenceTime.current = now;

      // 이미 처리 중이거나 세션이 없으면 건너뛰기
      console.log('isProcessing.value:::::', isProcessing.value);
      // if (isProcessing.value || !session) {

      console.log('Processing frame...');
      isProcessing.value = true;

      try {
        // 리사이징
        const resized = resize(frame, {
          scale: {
            width: MODEL_INPUT_WIDTH,
            height: MODEL_INPUT_HEIGHT,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        if (!resized) {
          console.error('리사이즈 실패');
          isProcessing.value = false;
          return;
        }

        // 일반 배열을 생성 (Worklet에서 Float32Array 생성 시 문제가 발생할 수 있음)
        const tensorDataArray: number[] = new Array(
          MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3,
        );

        // HWC to CHW 변환 및 정규화
        for (let i = 0; i < MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT; i++) {
          const r = resized[i * 3];
          const g = resized[i * 3 + 1];
          const b = resized[i * 3 + 2];

          tensorDataArray[i] = r / 255.0;
          tensorDataArray[i + MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT] =
            g / 255.0;
          tensorDataArray[i + 2 * MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT] =
            b / 255.0;
        }

        // 비동기 추론 실행 - 일반 배열을 전달
        console.log('비동기 추론 실행');
        runInference(tensorDataArray);
        console.log(
          '비동기 추론 요청 완료 (Worklet은 다음 라인을 계속 실행합니다)',
        );
      } catch (e) {
        const errorMessage =
          e instanceof Error ? `${e.name}: ${e.message}` : String(e);
        console.error(`전처리 오류: ${errorMessage}`);
        isProcessing.value = false;
      }
    },
    [session, resize, runInference], // ref는 dependency 배열에 추가할 필요가 없습니다.
  );

  if (!hasPermission)
    return (
      <View style={styles.container}>
        <Text style={styles.infoText}>카메라 권한이 없습니다.</Text>
      </View>
    );
  if (!device)
    return (
      <View style={styles.container}>
        <Text style={styles.infoText}>사용 가능한 카메라가 없습니다.</Text>
      </View>
    );

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        format={format}
        pixelFormat="yuv"
        fps={15}
      />
      {frameWidth > 0 && (
        <PoseOverlay
          poses={detectedPoses}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
        />
      )}
      <View style={styles.infoBox}>
        <Text style={styles.infoText}>
          {session ? '✅ 모델 로드 완료' : '⌛ 모델 로딩 중...'}
        </Text>
        <Text style={styles.infoText}>
          Frame: {frameWidth}x{frameHeight}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'black',
    justifyContent: 'center',
    alignItems: 'center',
  },
  infoBox: {
    position: 'absolute',
    top: 60,
    left: 20,
    backgroundColor: 'rgba(0,0,0,0.6)',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
  },
  infoText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
