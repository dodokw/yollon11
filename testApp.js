/**
 * App.tsx
 *
 * 이 파일은 애플리케이션의 메인 진입점입니다.
 * TFLite 모델을 사용하여 포즈 감지를 수행합니다.
 *
 * 주요 변경 사항:
 * - Python 코드와 동일한 '레터박스(Letterbox)' 전처리 방식을 frameProcessor에 직접 구현하여 모델 정확도 문제를 해결했습니다.
 * - CONFIDENCE_THRESHOLD를 0.5로 복구했습니다.
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

// 후처리 관련 상수
const CONFIDENCE_THRESHOLD = 0.5;

// test.py의 SSD 옵션을 상수로 정의
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

// --- 유틸리티 함수 (변경 없음) ---

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
): RotatedBbox => {
  const scale = 1.25;
  const { x: x0, y: y0 } = kpt0;
  const { x: x1, y: y1 } = kpt1;

  const dx = x1 - x0;
  const dy = y1 - y0;
  const angleRad = Math.atan2(dy, dx);

  const distance = Math.sqrt(dx * dx + dy * dy);
  const boxSize = distance * 2.0 * scale;

  const cx = x0;
  const cy = y0;
  const boxW = boxSize;
  const boxH = boxSize;

  const angle = angleRad - Math.PI / 2;
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);

  const halfW = boxW / 2;
  const halfH = boxH / 2;

  const corners = [
    { x: -halfW, y: -halfH },
    { x: halfW, y: -halfH },
    { x: halfW, y: halfH },
    { x: -halfW, y: halfH },
  ].map(p => ({
    x: cx + p.x * cos - p.y * sin,
    y: cy + p.x * sin + p.y * cos,
  }));

  return { corners };
};

const sigmoid = (x: number): number => {
  const clippedX = Math.max(-50, Math.min(x, 50));
  return 1 / (1 + Math.exp(-clippedX));
};

const processOutput = (output: Tensor[]): Pose[] => {
  const rawBoxes = output[0] as Float32Array;
  const rawScores = output[1] as Float32Array;

  if (!rawBoxes || !rawScores) {
    return [];
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

  if (maxScore > CONFIDENCE_THRESHOLD) {
    const anchor = anchors[bestIdx];
    const boxOffset = bestIdx * 12;

    const bbox = decodeBoxes(rawBoxes, anchor, boxOffset);
    const keypoints = decodeKeypoints(rawBoxes, anchor, boxOffset);

    if (keypoints.length >= 2) {
      const rotatedBbox = computeRotatedBboxFromKeypoints(
        keypoints[0],
        keypoints[1],
      );
      return [{ bbox, keypoints, score: maxScore, rotatedBbox }];
    }
  }

  return [];
};

// --- PoseOverlay 컴포넌트 및 하위 컴포넌트 (변경 없음) ---
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

  const detectedPoses = useSharedValue<Pose[]>([]);
  const isProcessing = useSharedValue(false);
  const lastInferenceTime = useRef(0);

  const runInference = useRunOnJS(
    async (frameDataAsArray: number[]) => {
      if (model == null) {
        isProcessing.value = false;
        return;
      }

      try {
        const frameData = new Float32Array(frameDataAsArray);
        const output = await model.run([frameData]);
        const poses = processOutput(output as Tensor[]);
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

      if (modelState !== 'loaded' || model == null || isProcessing.value) {
        return;
      }

      const now = Date.now();
      if (now - lastInferenceTime.current < 333) {
        return;
      }
      lastInferenceTime.current = now;
      isProcessing.value = true;

      try {
        // [수정] Python과 동일한 '레터박스' 전처리 로직 구현
        // 1. 중간 리사이즈 크기 계산 (비율 유지)
        const frameAspectRatio = frame.width / frame.height;
        const modelAspectRatio = MODEL_INPUT_WIDTH / MODEL_INPUT_HEIGHT;
        let newWidth = MODEL_INPUT_WIDTH;
        let newHeight = MODEL_INPUT_HEIGHT;

        if (frameAspectRatio > modelAspectRatio) {
          newHeight = Math.round(MODEL_INPUT_WIDTH / frameAspectRatio);
        } else {
          newWidth = Math.round(MODEL_INPUT_HEIGHT * frameAspectRatio);
        }

        // 2. 계산된 크기로 이미지 리사이즈
        const resized = resize(frame, {
          scale: {
            width: newWidth,
            height: newHeight,
          },
          pixelFormat: 'rgb',
          dataType: 'float32',
        });

        // 3. 224x224 캔버스 생성 및 검은색(-1.0)으로 채우기
        const canvas = new Float32Array(
          MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3,
        );
        canvas.fill(-1.0);

        // 4. 리사이즈된 이미지를 캔버스에 복사 (레터박싱) 및 정규화
        for (let y = 0; y < newHeight; y++) {
          for (let x = 0; x < newWidth; x++) {
            const sourceIndex = (y * newWidth + x) * 3;
            const destIndex = (y * MODEL_INPUT_WIDTH + x) * 3;

            canvas[destIndex] = (resized[sourceIndex] / 127.5) - 1.0; // R
            canvas[destIndex + 1] = (resized[sourceIndex + 1] / 127.5) - 1.0; // G
            canvas[destIndex + 2] = (resized[sourceIndex + 2] / 127.5) - 1.0; // B
          }
        }

        // 5. 레터박싱된 캔버스로 추론 실행
        runInference(Array.from(canvas));
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
      {frameWidth > 0 && (
        <PoseOverlay
          poses={detectedPoses}
          frameWidth={frameWidth}
          frameHeight={frameHeight}
        />
      )}
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
