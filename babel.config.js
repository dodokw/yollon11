module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [
    // 'vision-camera-resize-plugin'을 reanimated보다 위로 옮깁니다.
    'react-native-reanimated/plugin',
  ],
};
