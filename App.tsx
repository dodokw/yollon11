/**
 * App.tsx
 *
 * react-native-mediapipe 예제 코드를 기반으로 수정된 버전
 * 주요 수정사항:
 * 1. 기존 Pose Detection 로직 전체 제거
 * 2. Face Landmark Detection 로직으로 교체
 * 3. Skia를 사용한 랜드마크 드로잉 로직 추가
 * 4. 불필요한 import, 변수, 함수 정리
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  StyleSheet,
  Text,
  View,
  Pressable,
  ActivityIndicator,
} from 'react-native';
import {
  useCameraPermission,
  type CameraPosition,
} from 'react-native-vision-camera';
import {
  Delegate,
  MediapipeCamera,
  RunningMode,
  useFaceLandmarkDetection,
  faceLandmarkDetectionModuleConstants,
  type DetectionError,
  type FaceLandmarkDetectionResultBundle,
  type ViewCoordinator,
  type Landmark,
  type Dims,
} from 'react-native-mediapipe';
import Animated, { useSharedValue } from 'react-native-reanimated';
import { Canvas, Points, vec, type SkPoint } from '@shopify/react-native-skia';

// app-settings.js 또는 app-settings.tsx 파일이 있다고 가정합니다.
import { AppSettings, SettingsContext, useSettings } from './app-settings';

const MINIMUM_CONFIDENCE = 0.5;

// --- Drawing Component ---
// Skia를 사용하여 얼굴 랜드마크 연결선을 그리는 컴포넌트
const FaceDrawFrame: React.FC<{
  connections: Animated.SharedValue<SkPoint[]>;
  style: any;
}> = ({ connections, style }) => {
  return (
    <Canvas style={style}>
      <Points
        points={connections}
        pointMode="lines"
        color="#F95F48"
        strokeWidth={2}
      />
    </Canvas>
  );
};

// --- Permissions Component ---
// 카메라 권한이 없을 때 표시되는 컴포넌트
const NeedPermissions: React.FC<{ askForPermissions: () => void }> = ({
  askForPermissions,
}) => {
  return (
    <View style={styles.container}>
      <View style={styles.permissionsBox}>
        <Text style={styles.noPermsText}>카메라 사용을 허용해주세요</Text>
        <Text style={styles.permsInfoText}>
          얼굴 인식이 작동하려면 카메라 접근 권한이 필요합니다.
        </Text>
      </View>
      <Pressable style={styles.permsButton} onPress={askForPermissions}>
        <Text style={styles.permsButtonText}>허용</Text>
      </Pressable>
    </View>
  );
};

// --- Main Camera View Component ---
const CameraView = () => {
  const { settings } = useSettings();
  const { hasPermission, requestPermission } = useCameraPermission();
  const [activeCamera, setActiveCamera] = useState<CameraPosition>('front');

  const faceConnections = useSharedValue<SkPoint[]>([]);
  const isProcessing = useSharedValue(false);
  const { knownLandmarks } = faceLandmarkDetectionModuleConstants();

  const allConnections = React.useMemo(
    () => [
      ...knownLandmarks.lips,
      ...knownLandmarks.leftEye,
      ...knownLandmarks.rightEye,
      ...knownLandmarks.leftEyebrow,
      ...knownLandmarks.rightEyebrow,
      ...knownLandmarks.faceOval,
    ],
    [knownLandmarks],
  );

  const updateFaceConnections = useCallback(
    (newPoints: SkPoint[]) => {
      'worklet';
      faceConnections.value = newPoints;
    },
    [faceConnections],
  );

  const processFaceLandmarks = useCallback(
    (landmarks: Landmark[], frameDims: Dims, vc: ViewCoordinator) => {
      'worklet';
      if (isProcessing.value) return;

      try {
        isProcessing.value = true;
        const newLines: SkPoint[] = [];

        for (const { start, end } of allConnections) {
          if (!landmarks[start] || !landmarks[end]) continue;

          if (
            (landmarks[start].presence ?? 1) < MINIMUM_CONFIDENCE ||
            (landmarks[end].presence ?? 1) < MINIMUM_CONFIDENCE
          ) {
            continue;
          }

          const pt1 = vc.convertPoint(frameDims, landmarks[start]);
          const pt2 = vc.convertPoint(frameDims, landmarks[end]);

          if (
            !Number.isFinite(pt1.x) ||
            !Number.isFinite(pt1.y) ||
            !Number.isFinite(pt2.x) ||
            !Number.isFinite(pt2.y)
          ) {
            continue;
          }

          newLines.push(vec(pt1.x, pt1.y));
          newLines.push(vec(pt2.x, pt2.y));
        }
        updateFaceConnections(newLines);
      } finally {
        isProcessing.value = false;
      }
    },
    [allConnections, updateFaceConnections, isProcessing],
  );

  const onResults = React.useCallback(
    (results: FaceLandmarkDetectionResultBundle, vc: ViewCoordinator): void => {
      console.log(
        '******************isProcessing111******************************',
      );
      if (isProcessing.value) {
        console.log(
          '******************isProcessing222******************************',
        );
        return;
      }

      const frameDims = vc.getFrameDims(results);
      const landmarks = results.results[0]?.faceLandmarks[0] ?? [];
      console.log('******************landmarks******************************');

      if (landmarks.length > 0) {
        processFaceLandmarks(landmarks, frameDims, vc);
      } else {
        updateFaceConnections([]);
      }
    },
    [isProcessing, processFaceLandmarks, updateFaceConnections],
  );

  const onError = useCallback(
    (error: DetectionError): void => {
      console.error(`얼굴 감지 에러: ${JSON.stringify(error)}`);
      isProcessing.value = false;
    },
    [isProcessing],
  );

  // const faceDetection = useFaceLandmarkDetection(
  //   {
  //     onResults,
  //     onError,
  //   },
  //   RunningMode.LIVE_STREAM,
  //   'face_landmarker.task',
  //   // 예제 코드와 동일하게 4번째 인자로 옵션 객체를 전달합니다.
  //   {
  //     fpsMode: 30,
  //     delegate: settings.processor,
  //     mirrorMode: 'no-mirror',
  //   },
  // );
  const faceDetection = useFaceLandmarkDetection(
    onResults,
    onError,
    RunningMode.LIVE_STREAM,
    'face_landmarker.task',
    {
      fpsMode: 30,
      delegate: settings.processor,
      mirrorMode: 'no-mirror',
    },
  );
  // console.log('faceDetection:::', faceDetection);

  useEffect(() => {
    // 컴포넌트 언마운트 시 정리
    return () => {
      faceConnections.value = [];
      isProcessing.value = false;
    };
  }, [faceConnections, isProcessing]);

  if (!hasPermission) {
    return <NeedPermissions askForPermissions={requestPermission} />;
  }

  return (
    <View style={styles.container}>
      <MediapipeCamera
        style={styles.box}
        solution={faceDetection}
        activeCamera={activeCamera}
        resizeMode="cover"
      />
      {/* <FaceDrawFrame connections={faceConnections} style={styles.box} /> */}
      {/* <Pressable
        style={styles.cameraSwitchButton}
        onPress={() => setActiveCamera(prev => (prev === 'front' ? 'back' : 'front'))}
      >
        <Text style={styles.cameraSwitchButtonText}>카메라 전환</Text>
      </Pressable> */}
    </View>
  );
};

// App 컴포넌트는 Provider를 설정하는 역할만 담당
export default function App() {
  const [settings, setSettings] = useState<AppSettings>({
    maxResults: 5,
    threshold: 20,
    processor: Delegate.GPU,
    model: 'efficientdet-lite0',
  });

  return (
    <SettingsContext.Provider value={{ settings, setSettings }}>
      <CameraView />
    </SettingsContext.Provider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: 'blue',
    alignItems: 'center',
    justifyContent: 'center',
  },
  box: {
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
  },
  permissionsBox: {
    backgroundColor: '#F3F3F3',
    padding: 20,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#CCCACA',
    marginBottom: 20,
    marginHorizontal: 20,
  },
  noPermsText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'black',
    textAlign: 'center',
  },
  permsInfoText: {
    fontSize: 15,
    color: 'black',
    marginTop: 12,
    textAlign: 'center',
  },
  permsButton: {
    paddingVertical: 15,
    paddingHorizontal: 25,
    backgroundColor: '#F95F48',
    borderRadius: 5,
    margin: 15,
  },
  permsButtonText: {
    fontSize: 17,
    color: 'white',
    fontWeight: 'bold',
  },
  cameraSwitchButton: {
    position: 'absolute',
    padding: 10,
    backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 20,
    top: 60,
    right: 20,
  },
  cameraSwitchButtonText: {
    color: 'white',
    fontSize: 16,
  },
});
