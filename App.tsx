/**
 * App.tsx
 *
 * PoseOverlay 노출 문제 해결 및 정밀도 개선 버전
 */
import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from 'react';
import {
  StyleSheet,
  Text,
  View,
  Dimensions,
  Platform,
  ActivityIndicator,
} from 'react-native';
import {
  Camera,
  Frame,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from 'react-native-vision-camera';
import { Svg, Circle, Polygon, Line } from 'react-native-svg';
import Animated, {
  useSharedValue,
  useAnimatedProps,
} from 'react-native-reanimated';
import {
  Tensor,
  TensorflowModel,
  useTensorflowModel,
} from 'react-native-fast-tflite';
import { useResizePlugin } from 'vision-camera-resize-plugin';
import { useRunOnJS } from 'react-native-worklets-core';

// 화면 크기 가져오기
const { width: screenWidth, height: screenHeight } = Dimensions.get('window');

// 모델 관련 상수
const MODEL_INPUT_WIDTH = 224;
const MODEL_INPUT_HEIGHT = 224;

// 후처리 관련 상수 - 파이썬과 동일하게 설정
const CONFIDENCE_THRESHOLD = 0.5;

// SSD 옵션을 상수로 정의
const SSD_OPTIONS = {
  num_layers: 5,
  input_size_height: 224,
  input_size_width: 224,
  anchor_offset_x: 0.5,
  anchor_offset_y: 0.5,
  min_scale: 0.1484375,
  max_scale: 0.75,
  strides: [8, 16, 32, 32, 32],
  fixed_anchor_size: true,
};

// 타입 정의
interface Keypoint {
  x: number;
  y: number;
}

interface Pose {
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
  keypoints: Keypoint[];
  score: number;
  rotatedBbox: RotatedBbox;
}

interface RotatedBbox {
  corners: { x: number; y: number }[];
}

interface ProcessedOutput {
  poses: Pose[];
  bestIdx: number;
  maxScore: number;
}

// Reanimated와 SVG를 연결하기 위한 컴포넌트 생성
const AnimatedCircle = Animated.createAnimatedComponent(Circle);
const AnimatedPolygon = Animated.createAnimatedComponent(Polygon);

// --- 유틸리티 함수 ---

const calculateScale = (
  minScale: number,
  maxScale: number,
  strideIndex: number,
  numStrides: number,
): number => {
  if (numStrides === 1) {
    return (minScale + maxScale) * 0.5;
  }
  return minScale + ((maxScale - minScale) * strideIndex) / (numStrides - 1);
};

const ssdGenerateAnchors = (): number[][] => {
  const anchors: number[][] = [];
  const {
    num_layers,
    strides,
    input_size_height,
    input_size_width,
    anchor_offset_x,
    anchor_offset_y,
    min_scale,
    max_scale,
    fixed_anchor_size,
  } = SSD_OPTIONS;

  const scales = Array.from({ length: num_layers }, (_, i) =>
    calculateScale(min_scale, max_scale, i, num_layers),
  );

  for (let layer_id = 0; layer_id < num_layers; layer_id++) {
    const stride = strides[layer_id];
    const feature_map_height = Math.ceil(input_size_height / stride);
    const feature_map_width = Math.ceil(input_size_width / stride);

    for (let y = 0; y < feature_map_height; y++) {
      const y_center = (y + anchor_offset_y) / feature_map_height;
      for (let x = 0; x < feature_map_width; x++) {
        const x_center = (x + anchor_offset_x) / feature_map_width;
        if (fixed_anchor_size) {
          anchors.push([x_center, y_center, 1.0, 1.0]);
        }
      }
    }
  }
  return anchors;
};

const anchors = ssdGenerateAnchors();

const decodeBoxes = (
  rawBoxes: Float32Array,
  anchor: number[],
  boxOffset: number,
): Pose['bbox'] => {
  const scale = 224.0;
  const [anchorX, anchorY, anchorW, anchorH] = anchor;

  const x_center = (rawBoxes[boxOffset] / scale) * anchorW + anchorX;
  const y_center = (rawBoxes[boxOffset + 1] / scale) * anchorH + anchorY;
  const w = (rawBoxes[boxOffset + 2] / scale) * anchorW;
  const h = (rawBoxes[boxOffset + 3] / scale) * anchorH;

  return {
    xmin: x_center - w / 2,
    ymin: y_center - h / 2,
    xmax: x_center + w / 2,
    ymax: y_center + h / 2,
  };
};

const decodeKeypoints = (
  rawBoxes: Float32Array,
  anchor: number[],
  boxOffset: number,
): Keypoint[] => {
  const scale = 224.0;
  const [anchorX, anchorY, anchorW, anchorH] = anchor;
  const keypoints: Keypoint[] = [];

  for (let i = 0; i < 4; i++) {
    const offset = boxOffset + 4 + i * 2;
    let x = (rawBoxes[offset] / scale) * anchorW + anchorX;
    let y = (rawBoxes[offset + 1] / scale) * anchorH + anchorY;

    x = Math.max(0, Math.min(1, x));
    y = Math.max(0, Math.min(1, y));

    keypoints.push({ x, y });
  }
  return keypoints;
};

const computeRotatedBboxFromKeypoints = (
  kpt0: Keypoint,
  kpt1: Keypoint,
  imageW: number = 224,
  imageH: number = 224,
  scale: number = 1.25,
): RotatedBbox => {
  const { x: x0, y: y0 } = kpt0;
  const { x: x1, y: y1 } = kpt1;

  const cx = x0 * imageW;
  const cy = y0 * imageH;
  const dx = x1 * imageW - cx;
  const dy = -(y1 * imageH - cy);

  const angle = Math.atan2(dy, dx) * (180 / Math.PI) - 90;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const boxSize = distance * 2.0 * scale;

  const angleRad = (angle * Math.PI) / 180;
  const cos = Math.cos(angleRad);
  const sin = Math.sin(angleRad);

  const halfSize = boxSize / 2;
  const corners = [
    { x: -halfSize, y: -halfSize },
    { x: halfSize, y: -halfSize },
    { x: halfSize, y: halfSize },
    { x: -halfSize, y: halfSize },
  ].map(p => ({
    x: (cx + p.x * cos - p.y * sin) / imageW,
    y: (cy + p.x * sin + p.y * cos) / imageH,
  }));

  return { corners };
};

const sigmoid = (x: number): number => {
  const clippedX = Math.max(-50, Math.min(x, 50));
  return 1 / (1 + Math.exp(-clippedX));
};

const processOutput = (output: (Tensor | Float32Array)[]): ProcessedOutput => {
  const rawBoxes = output[0] as Float32Array;
  const rawScores = output[1] as Float32Array;

  if (!rawBoxes || !rawScores) {
    console.log('!rawBoxes || !rawScores');
    return { poses: [], bestIdx: -1, maxScore: -1 };
  }

  let bestIdx = -1;
  let maxScore = -1;
  for (let i = 0; i < rawScores.length; i++) {
    const score = sigmoid(rawScores[i]);
    if (score > maxScore) {
      maxScore = score;
      bestIdx = i;
    }
  }

  // console.log('maxScore:::', maxScore);
  // console.log('bestIdx:::', bestIdx);

  if (maxScore > CONFIDENCE_THRESHOLD) {
    const anchor = anchors[bestIdx];
    const boxOffset = bestIdx * 12;

    const bbox = decodeBoxes(rawBoxes, anchor, boxOffset);
    const keypoints = decodeKeypoints(rawBoxes, anchor, boxOffset);

    if (keypoints.length >= 2) {
      const rotatedBbox = computeRotatedBboxFromKeypoints(
        keypoints[0],
        keypoints[1],
        MODEL_INPUT_WIDTH,
        MODEL_INPUT_HEIGHT,
      );

      return {
        poses: [{ bbox, keypoints, score: maxScore, rotatedBbox }],
        bestIdx,
        maxScore,
      };
    }
  }

  return { poses: [], bestIdx, maxScore };
};

// --- Static Drawing Components ---
// NOTE: For better code structure, these are defined before being used in PoseOverlay.

const StaticKeypoint = ({ x, y }: { x: number; y: number }) => {
  return (
    <Circle cx={x} cy={y} r={8} fill="red" stroke="white" strokeWidth={2} />
  );
};

const StaticRotatedBox = ({
  corners,
}: {
  corners: { x: number; y: number }[];
}) => {
  const points = corners.map(p => `${p.x},${p.y}`).join(' ');

  return (
    <Polygon points={points} fill="none" stroke="orange" strokeWidth={3} />
  );
};

const StaticBoundingBox = ({
  bbox,
}: {
  bbox: { xmin: number; ymin: number; xmax: number; ymax: number };
}) => {
  const { xmin, ymin, xmax, ymax } = bbox;
  return (
    <Polygon
      points={`${xmin},${ymin} ${xmax},${ymin} ${xmax},${ymax} ${xmin},${ymax}`}
      fill="none"
      stroke="blue"
      strokeWidth={2}
    />
  );
};

// [수정됨] PoseOverlay 컴포넌트
const PoseOverlay = ({
  poses,
  frameWidth,
  frameHeight,
}: {
  poses: Pose[];
  frameWidth: number;
  frameHeight: number;
}) => {
  // 프레임 크기가 유효하지 않으면 렌더링하지 않음 (0으로 나누기 방지)
  if (!frameWidth || !frameHeight) {
    return null;
  }

  // 1. 화면 표시 영역 계산 ('contain' 모드)
  // 카메라 프레임이 화면에 어떻게 표시되는지 계산합니다.
  const screenAspectRatio = screenWidth / screenHeight;
  const frameAspectRatio = frameWidth / frameHeight;

  let displayWidth, displayHeight, offsetX, offsetY;
  if (frameAspectRatio > screenAspectRatio) {
    // 프레임이 화면보다 넓은 경우 (레터박스)
    displayWidth = screenWidth;
    displayHeight = screenWidth / frameAspectRatio;
    offsetX = 0;
    offsetY = (screenHeight - displayHeight) / 2;
  } else {
    // 프레임이 화면보다 높은 경우 (필러박스)
    displayWidth = screenHeight * frameAspectRatio;
    displayHeight = screenHeight;
    offsetX = (screenWidth - displayWidth) / 2;
    offsetY = 0;
  }

  // 2. 전처리 보정 계수 계산
  // 모델 입력(정사각형)에 맞추기 위해 프레임에 적용된 레터박싱/필러박싱을 보정합니다.
  const modelAspectRatio = MODEL_INPUT_WIDTH / MODEL_INPUT_HEIGHT; // 1.0

  let correctionX = 1;
  let correctionY = 1;
  if (frameAspectRatio > modelAspectRatio) {
    // 원본 프레임이 모델 입력보다 넓음 -> Y축 좌표가 압축됨
    correctionY = frameAspectRatio / modelAspectRatio;
  } else {
    // 원본 프레임이 모델 입력보다 높음 -> X축 좌표가 압축됨
    correctionX = modelAspectRatio / frameAspectRatio;
  }

  // console.log('PoseOverlay 렌더링:', {
  //   posesLength: poses.length,
  //   displayWidth,
  //   displayHeight,
  //   offsetX,
  //   offsetY,
  //   correctionX,
  //   correctionY,
  // });

  if (poses.length === 0) {
    return null;
  }

  // 3. 좌표 변환 함수
  // 모델 좌표를 화면 좌표로 변환합니다.
  const transformX = (x: number) => x * correctionX * displayWidth + offsetX;
  const transformY = (y: number) => y * correctionY * displayHeight + offsetY;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      <Svg
        width={screenWidth}
        height={screenHeight}
        style={StyleSheet.absoluteFill}
      >
        {poses.map((pose, poseIndex) => (
          <React.Fragment key={`pose-${poseIndex}`}>
            {/* 키포인트 렌더링 */}
            {pose.keypoints.map((keypoint, keypointIndex) => {
              const screenX = transformX(keypoint.x);
              const screenY = transformY(keypoint.y);
              return (
                <StaticKeypoint
                  key={`kp-${poseIndex}-${keypointIndex}`}
                  x={screenX}
                  y={screenY}
                />
              );
            })}

            {/* 바운딩 박스 렌더링 */}
            <StaticBoundingBox
              bbox={{
                xmin: transformX(pose.bbox.xmin),
                ymin: transformY(pose.bbox.ymin),
                xmax: transformX(pose.bbox.xmax),
                ymax: transformY(pose.bbox.ymax),
              }}
            />

            {/* 회전된 박스 렌더링 */}
            {pose.rotatedBbox && (
              <StaticRotatedBox
                corners={pose.rotatedBbox.corners.map(corner => ({
                  x: transformX(corner.x),
                  y: transformY(corner.y),
                }))}
              />
            )}
          </React.Fragment>
        ))}
      </Svg>
    </View>
  );
};

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const { resize } = useResizePlugin();
  const [timer, setTimer] = useState<number>(0);
  const [modelRunTime, setModelRunTime] = useState<number>(0);
  const [preProcessTime, setPreProcessTime] = useState<number>(0);
  const [postProcessTime, setPostProcessTime] = useState<number>(0);
  const [commTime, setCommTime] = useState<number>(0);
  const [bestIdx, setBestIdx] = useState<number>(-1);
  const [maxScore, setMaxScore] = useState<number>(-1);

  // SharedValue 대신 일반 state 사용
  const [detectedPoses, setDetectedPoses] = useState<Pose[]>([]);

  const { state: modelState, model } = useTensorflowModel(
    require('./assets/pose_detection.tflite'),
  );

  useEffect(() => {
    requestPermission();
  }, [requestPermission]);

  useEffect(() => {
    if (modelState === 'loaded' && model) {
      console.log('--- TFLite Model Details ---');
      console.log('Input Tensors:', JSON.stringify(model.inputs, null, 2));
      console.log('Output Tensors:', JSON.stringify(model.outputs, null, 2));
      console.log('--------------------------');
    }
  }, [modelState, model]);

  const format = useMemo(() => {
    if (!device?.formats) return undefined;
    const allFormats = device.formats;
    const filteredFormats = allFormats.filter(
      f => f.videoWidth <= 1920 && f.maxFps >= 15,
    );
    if (filteredFormats.length > 0) {
      filteredFormats.sort((a, b) => a.videoWidth - b.videoWidth);
      return filteredFormats[0];
    }
    return device.formats[0];
  }, [device?.formats]);

  const frameWidth = format?.videoWidth ?? 0;
  const frameHeight = format?.videoHeight ?? 0;

  const isProcessing = useSharedValue(false);
  const lastInferenceTime = useRef(0);

  const runInference = useRunOnJS(
    async (
      frameDataAsArray: number[],
      originalWidth: number,
      originalHeight: number,
      startTime: number,
      preProcessEndTime: number,
    ) => {
      const jsThreadStartTime = Date.now();
      if (model == null) {
        isProcessing.value = false;
        return;
      }

      try {
        // 워크릿 종료 시점과 JS 스레드 시작 시점의 차이를 계산
        setCommTime(jsThreadStartTime - preProcessEndTime);
        setPreProcessTime(preProcessEndTime - startTime);

        const frameData = new Float32Array(frameDataAsArray);

        const modelStartTime = Date.now();
        const output = await model.run([frameData]);
        const modelEndTime = Date.now();
        setModelRunTime(modelEndTime - modelStartTime);

        const postProcessStartTime = modelEndTime;
        const {
          poses,
          bestIdx: newBestIdx,
          maxScore: newMaxScore,
        } = processOutput(output);
        const postProcessEndTime = Date.now();
        setPostProcessTime(postProcessEndTime - postProcessStartTime);

        setTimer(postProcessEndTime - startTime);

        // console.log('*************poseCameOut*************');
        // console.log(JSON.stringify(poses));
        // console.log('*************poseCameOut*************');

        // SharedValue 대신 setState 사용
        setDetectedPoses(poses);
        setBestIdx(newBestIdx);
        setMaxScore(newMaxScore);
      } catch (e) {
        console.error('TFLite 추론 오류:', e);
      } finally {
        isProcessing.value = false;
      }
    },
    [model],
  );

  const frameProcessor = useFrameProcessor(
    (frame: Frame) => {
      'worklet';

      if (modelState !== 'loaded' || model == null) {
        return;
      }

      const now = Date.now();
      if (now - lastInferenceTime.current < 1000) {
        // 1초마다 처리로 변경
        return;
      }
      lastInferenceTime.current = now;
      isProcessing.value = true;

      try {
        const startTime = Date.now();
        const targetW = MODEL_INPUT_WIDTH;
        const targetH = MODEL_INPUT_HEIGHT;

        const scale = Math.min(targetW / frame.width, targetH / frame.height);
        const resizedW = Math.floor(frame.width * scale);
        const resizedH = Math.floor(frame.height * scale);

        const resized = resize(frame, {
          scale: {
            width: resizedW,
            height: resizedH,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        const inputTensor = new Uint8Array(targetW * targetH * 3);

        for (let y = 0; y < resizedH; y++) {
          for (let x = 0; x < resizedW; x++) {
            const srcIdx = (y * resizedW + x) * 3;
            const dstIdx = (y * targetW + x) * 3;
            inputTensor[dstIdx] = resized[srcIdx];
            inputTensor[dstIdx + 1] = resized[srcIdx + 1];
            inputTensor[dstIdx + 2] = resized[srcIdx + 2];
          }
        }

        const normalizedTensor = new Float32Array(targetW * targetH * 3);
        for (let i = 0; i < inputTensor.length; i++) {
          normalizedTensor[i] = inputTensor[i] / 127.5 - 1.0;
        }

        const preProcessEndTime = Date.now();

        runInference(
          Array.from(normalizedTensor),
          frame.width,
          frame.height,
          startTime,
          preProcessEndTime,
        );
      } catch (e) {
        const errorMessage =
          e instanceof Error ? `${e.name}: ${e.message}` : String(e);
        console.error(`프레임 전처리 오류: ${errorMessage}`);
        isProcessing.value = false;
      }
    },
    [modelState, model, resize, runInference],
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

      {/* 수정된 PoseOverlay 사용 */}
      <PoseOverlay
        poses={detectedPoses}
        frameWidth={frameWidth}
        frameHeight={frameHeight}
      />

      <View style={styles.infoBox}>
        {modelState === 'loading' && (
          <>
            <ActivityIndicator size="small" color="white" />
            <Text style={styles.infoText}>⌛ Model on Loading...</Text>
          </>
        )}
        {modelState === 'loaded' && (
          <Text style={styles.infoText}>✅ Model Load</Text>
        )}
        {modelState === 'error' && (
          <Text style={styles.infoText}>❌ Model Load</Text>
        )}
        <Text style={styles.infoText}>
          Frame: {frameWidth}x{frameHeight}
        </Text>
        <Text style={styles.infoText}>Total Time: {timer}ms</Text>
        <Text style={styles.infoText}>
          - Pre-processing: {preProcessTime}ms
        </Text>
        <Text style={styles.infoText}>- Bridge & Queue: {commTime}ms</Text>
        <Text style={styles.infoText}>- Model Inference: {modelRunTime}ms</Text>
        <Text style={styles.infoText}>
          - Post-processing: {postProcessTime}ms
        </Text>
        <Text style={styles.infoText}>
          Detected Poses: {detectedPoses.length}
        </Text>
        <Text style={styles.infoText}>Best Index: {bestIdx}</Text>
        <Text style={styles.infoText}>
          Max Score: {maxScore > 0 ? maxScore.toFixed(4) : 'N/A'}
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
    gap: 8,
  },
  infoText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
