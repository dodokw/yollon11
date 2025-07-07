/**
 * App.tsx
 *
 * 파이썬 코드와 동일한 전처리 로직을 적용한 버전
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
    const x = (rawBoxes[offset] / scale) * anchorW + anchorX;
    const y = (rawBoxes[offset + 1] / scale) * anchorH + anchorY;
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
  const dy = -(y1 * imageH - cy); // Y축 반전 (파이썬과 동일)

  const angle = Math.atan2(dy, dx) * (180 / Math.PI) - 90;
  const distance = Math.sqrt(dx * dx + dy * dy);
  const boxSize = distance * 2.0 * scale;

  // 회전된 박스의 코너 계산
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

// sigmoid 함수 (파이썬과 동일한 클리핑 적용)
const sigmoid = (x: number): number => {
  const clippedX = Math.max(-50, Math.min(x, 50));
  return 1 / (1 + Math.exp(-clippedX));
};

// 후처리 함수
const processOutput = (output: { [key: string]: Tensor }): Pose[] => {
  const rawBoxes = output[0] as Float32Array;
  const rawScores = output[1] as Float32Array;

  if (!rawBoxes || !rawScores) {
    console.log('!rawBoxes || !rawScores');
    return [];
  }

  // 가장 높은 점수의 인덱스를 찾습니다.
  let bestIdx = -1;
  let maxScore = -1;
  for (let i = 0; i < rawScores.length; i++) {
    const score = sigmoid(rawScores[i]);
    if (score > maxScore) {
      maxScore = score;
      bestIdx = i;
    }
  }

  console.log('maxScore:::', maxScore);
  console.log('bestIdx:::', bestIdx);

  // 임계값을 넘는 경우에만 포즈를 계산합니다.
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

      return [{ bbox, keypoints, score: maxScore, rotatedBbox }];
    }
  }

  return [];
};

// --- PoseOverlay 컴포넌트들 (기존과 동일) ---
const AnimatedKeypoint = React.memo(
  ({
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
    const animatedProps = useAnimatedProps(() => {
      const pose = poses.value[poseIndex];
      if (!pose) return { display: 'none' };

      const keypoint = pose.keypoints[keypointIndex];
      if (!keypoint) return { display: 'none' };
      return {
        cx: keypoint.x * scaleX + offsetX,
        cy: keypoint.y * scaleY + offsetY,
        r: 5,
        fill: 'red',
        display: 'flex',
      };
    });
    return <AnimatedCircle animatedProps={animatedProps} />;
  },
);

const AnimatedRotatedBox = React.memo(
  ({
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
    const animatedProps = useAnimatedProps(() => {
      const pose = poses.value[poseIndex];
      if (!pose || !pose.rotatedBbox) return { display: 'none' };

      const points = pose.rotatedBbox.corners
        .map(p => `${p.x * scaleX + offsetX},${p.y * scaleY + offsetY}`)
        .join(' ');

      return { points, display: 'flex' };
    });

    return (
      <AnimatedPolygon
        animatedProps={animatedProps}
        fill="none"
        stroke="orange"
        strokeWidth="2"
      />
    );
  },
);

const SinglePose = React.memo(
  ({
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
        {Array.from({ length: 4 }).map((_, keypointIndex) => (
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
        <AnimatedRotatedBox
          poses={poses}
          poseIndex={poseIndex}
          scaleX={scaleX}
          scaleY={scaleY}
          offsetX={offsetX}
          offsetY={offsetY}
        />
      </React.Fragment>
    );
  },
);

const PoseOverlay = ({
  poses,
  frameWidth,
  frameHeight,
}: {
  poses: Animated.SharedValue<Pose[]>;
  frameWidth: number;
  frameHeight: number;
}) => {
  const screenAspectRatio = screenWidth / screenHeight;
  const frameAspectRatio = frameWidth / frameHeight;

  let offsetX = 0;
  let offsetY = 0;
  let previewWidth = screenWidth;
  let previewHeight = screenHeight;

  if (frameAspectRatio > screenAspectRatio) {
    previewWidth = screenHeight * frameAspectRatio;
    offsetX = (screenWidth - previewWidth) / 2;
  } else {
    previewHeight = screenWidth / frameAspectRatio;
    offsetY = (screenHeight - previewHeight) / 2;
  }

  const scaleX = previewWidth / MODEL_INPUT_WIDTH;
  const scaleY = previewHeight / MODEL_INPUT_HEIGHT;

  return (
    <Svg
      width={screenWidth}
      height={screenHeight}
      style={StyleSheet.absoluteFill}
    >
      <SinglePose
        poses={poses}
        poseIndex={0}
        scaleX={scaleX}
        scaleY={scaleY}
        offsetX={offsetX}
        offsetY={offsetY}
      />
    </Svg>
  );
};

export default function App() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('front');
  const { resize } = useResizePlugin();
  const [timer, setTimer] = useState<number>(0); //ms
  const [modelRunTime, setModelRunTime] = useState<number>(0); //ms

  const { state: modelState, model } = useTensorflowModel(
    require('./assets/pose_detection.tflite'),
  );

  useEffect(() => {
    requestPermission();
  }, [requestPermission]);

  // 모델 로드 시 입/출력 정보 확인
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

  const detectedPoses = useSharedValue<Pose[]>([]);
  const isProcessing = useSharedValue(false);
  const lastInferenceTime = useRef(0);

  const runInference = useRunOnJS(
    async (
      frameDataAsArray: number[],
      originalWidth: number,
      originalHeight: number,
      startTime: number,
    ) => {
      if (model == null) {
        isProcessing.value = false;
        return;
      }

      try {
        // 파이썬과 동일한 전처리 적용
        const targetW = MODEL_INPUT_WIDTH;
        const targetH = MODEL_INPUT_HEIGHT;

        // aspect ratio 유지하면서 스케일 계산
        const scale = Math.min(
          targetW / originalWidth,
          targetH / originalHeight,
        );
        const resizedW = Math.floor(originalWidth * scale);
        const resizedH = Math.floor(originalHeight * scale);

        console.log(
          `Original: ${originalWidth}x${originalHeight}, Scale: ${scale}, Resized: ${resizedW}x${resizedH}`,
        );

        const frameData = new Float32Array(frameDataAsArray);

        const modelStartTime = Date.now();
        const output = await model.run([frameData]);
        const modelEndTime = Date.now();
        setModelRunTime(modelEndTime - modelStartTime);

        const poses = processOutput(output);

        const endTime = Date.now();
        setTimer(endTime - startTime);

        console.log('*************poseCameOut*************');
        console.log(JSON.stringify(poses));
        console.log('*************poseCameOut*************');

        detectedPoses.value = poses;
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
        // 1초마다 처리
        return;
      }
      lastInferenceTime.current = now;
      isProcessing.value = true;

      try {
        const startTime = Date.now();
        // 파이썬과 동일한 전처리 방식 적용
        const targetW = MODEL_INPUT_WIDTH;
        const targetH = MODEL_INPUT_HEIGHT;

        // aspect ratio 유지하면서 스케일 계산
        const scale = Math.min(targetW / frame.width, targetH / frame.height);
        const resizedW = Math.floor(frame.width * scale);
        const resizedH = Math.floor(frame.height * scale);

        // 리사이징 (aspect ratio 유지)
        const resized = resize(frame, {
          scale: {
            width: resizedW,
            height: resizedH,
          },
          pixelFormat: 'rgb',
          dataType: 'uint8',
        });

        // 224x224 텐서 생성 (0으로 패딩)
        const inputTensor = new Uint8Array(targetW * targetH * 3);

        // 리사이징된 이미지를 좌상단에 배치
        for (let y = 0; y < resizedH; y++) {
          for (let x = 0; x < resizedW; x++) {
            const srcIdx = (y * resizedW + x) * 3;
            const dstIdx = (y * targetW + x) * 3;
            inputTensor[dstIdx] = resized[srcIdx]; // R
            inputTensor[dstIdx + 1] = resized[srcIdx + 1]; // G
            inputTensor[dstIdx + 2] = resized[srcIdx + 2]; // B
          }
        }

        // 정규화 (0-255 -> -1~1)
        const normalizedTensor = new Float32Array(targetW * targetH * 3);
        for (let i = 0; i < inputTensor.length; i++) {
          normalizedTensor[i] = inputTensor[i] / 127.5 - 1.0;
        }

        runInference(
          Array.from(normalizedTensor),
          frame.width,
          frame.height,
          startTime,
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
      {/* {frameWidth > 0 && (
        <PoseOverlay
          poses={detectedPoses}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
        />
      )} */}
      <View style={styles.infoBox}>
        {modelState === 'loading' && (
          <>
            <ActivityIndicator size="small" color="white" />
            <Text style={styles.infoText}>⌛ 모델 로딩 중...</Text>
          </>
        )}
        {modelState === 'loaded' && (
          <Text style={styles.infoText}>✅ 모델 로드 완료</Text>
        )}
        {modelState === 'error' && (
          <Text style={styles.infoText}>❌ 모델 로드 실패</Text>
        )}
        <Text style={styles.infoText}>
          Frame: {frameWidth}x{frameHeight}
        </Text>
        <Text style={styles.infoText}>
          preprocessing Time: {timer - modelRunTime}ms
        </Text>
        <Text style={styles.infoText}>Model Inference: {modelRunTime}ms</Text>
        <Text style={styles.infoText}>Total Time: {timer}ms</Text>
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
