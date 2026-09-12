#pragma once
#include <cmath>
#include <cstdint>

// Actuator intensity, not measured velocity. Fractional state preserves small
// increments. The terminal clamp prevents overshoot; emergency stops bypass it.
struct MotionProfile {
  float value = 0.0f;
  float rate = 0.0f;
  static float clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
  }
  void reset() { value = rate = 0.0f; }
  int8_t step(float target, float dt) {
    dt = clamp(dt, 0.0f, 0.02f);
    if (dt == 0.0f) return static_cast<int8_t>(std::lround(value));
    float error = target - value;
    float desired = clamp(error * 8.0f, -100.0f, 100.0f);
    rate += clamp(desired - rate, -600.0f * dt, 600.0f * dt);
    float change = rate * dt;
    if ((error >= 0.0f && change >= error) || (error <= 0.0f && change <= error)) {
      value = target;
      rate = 0.0f;
    } else {
      value = clamp(value + change, -127.0f, 127.0f);
    }
    return static_cast<int8_t>(std::lround(value));
  }
};
