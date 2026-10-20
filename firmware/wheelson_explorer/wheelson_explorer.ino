/**
 * wheelson_explorer.ino
 *
 * CircuitMess Wheelson — Autonomous Explorer Firmware
 * ─────────────────────────────────────────────────────
 * Exposes a minimal REST API over Wi-Fi so the Python middleware
 * can see through the robot's eyes and steer its wheels.
 *
 * Endpoints
 *   GET  /camera  → JPEG frame; X-Distance-CM response header
 *   GET  /status  → JSON { distance_cm, ip }
 *   POST /move    → JSON { action: "forward|backward|left|right|stop",
 *                          duration_ms: <int> }
 *
 * Motor control talks directly to the Nuvoton coprocessor over I2C
 * (address 0x38) using the same byte protocol as WheelsonMotor.cpp.
 * No CircuitOS dependency required — just the standard ESP32 Arduino
 * core and the ArduinoJson library.
 *
 * Distance sensor: HC-SR04 wiring is retained for compatibility, but
 * ultrasonic echo is not used for safety decisions in this build.
 *
 * ─── Arduino IDE setup ───────────────────────────────────────────────
 *  Board      : ESP32 Dev Module
 *  Partition  : Huge APP (3MB No OTA/1MB SPIFFS)
 *  PSRAM      : Enabled
 *  No external libraries required — uses ESP32 Arduino core only.
 */

#include "esp_camera.h"
#include "img_converters.h"
#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>
#include <Wire.h>
#include "motion_profile.h"
#include "freertos/semphr.h"
#include "freertos/task.h"

// ─── USER CONFIGURATION ──────────────────────────────────────────────────────

#if __has_include("secrets.h")
#include "secrets.h"
#endif

#ifndef WIFI_SSID
#define WIFI_SSID "WIFI_SSID"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "WIFI_PASSWORD"
#endif

// HC-SR04 distance sensor (must stay on 13/12 due to hardware limits)
#define TRIG_PIN 13
#define ECHO_PIN 12

// Distance echo sensor is currently disabled in runtime safety decisions.
#define ULTRASONIC_ENABLED false

// Motor drive intensity: 0–127 (signed int8_t range used by Nuvoton)
// SPEED VARIABLES
int8_t currentSpeed = 40;
const float FORWARD_SPEED_SCALE = 0.50f;
const int8_t SPEED_SLOW = 25;
const int8_t SPEED_MEDIUM = 40;
const int8_t SPEED_FAST = 70;

// MOVEMENT STATE
String currentMoveAction = "stop";
String commandedAction = "stop";
bool isMovingIndefinitely = false;
unsigned long lastDistanceCheckTime = 0;
const int DISTANCE_CHECK_INTERVAL_MS =
    150; // Increased to 150ms for reliable echoes
const unsigned long COMMAND_LEASE_TIMEOUT_MS = 750;

// ON-DEVICE NAVIGATION STATE (reactive safety only; no tactical recovery FSM)
enum NavState { NAV_HOLD, NAV_CRUISE };
NavState navState = NAV_HOLD;

// ─── NUVOTON I2C MOTOR CONTROLLER ────────────────────────────────────────────
// Protocol reverse-engineered from
// Wheelson-Library/src/Nuvoton/WheelsonMotor.cpp
#define NUVOTON_ADDR 0x38       // Nuvoton co-processor I2C address
#define MOTOR_SET_BYTE 0x30     // Command byte to set motor intensity
#define HEADLIGHT_SET_BYTE 0x22 // Command byte to set headlight brightness
#define I2C_SDA 14              // Wheelson I2C bus (from Pins.hpp)
#define I2C_SCL 15

// ─── VISUAL TELEMETRY VARIABLES ──────────────────────────────────────────────
bool telemetryObstructed = false;
String telemetryBrightness = "Normal";
String telemetryColorHex = "#000000";
float telemetryObstacleLeftRatio = 0.0f;
float telemetryObstacleRightRatio = 0.0f;
uint8_t *visual_rgb_buf = nullptr;

// Smoothed (EMA) obstacle ratios and latched obstruction with hysteresis.
// Raw per-frame ratios are noisy; decisions use the filtered values.
#define OBSTACLE_EMA_ALPHA 0.35f
float emaObstacleLeftRatio = 0.0f;
float emaObstacleRightRatio = 0.0f;
bool emaSeeded = false;
float obstacleEnterRatio = 0.40f; // settable via /move "obstacle_threshold" (thousandths)
const float OBSTACLE_EXIT_RATIO_FACTOR = 0.75f; // exit at enter*0.75

// Rolling floor color reference: only updated while the frame looks clear so
// a passing obstacle never poisons the reference.
int refFloorR = -1;
int refFloorG = -1;
int refFloorB = -1;
static const int FLOOR_REF_ALPHA_NUM = 1;
static const int FLOOR_REF_ALPHA_DEN = 8;

// Last accepted command metadata (from middleware authority)
String activeCommandId = "boot";
String activeCommandSource = "bootstrap";
String activeCommandMode = "idle";

// Sticky one-shot safety latch consumed by /move responses
bool safetyStopLatched = false;
String safetyStopReason = "";

// Dynamic visual safety configuration
bool visualObstacleSafetyEnabled = true;

// Motor IDs (from Wheelson-Library/src/Pins.hpp)
#define MOTOR_FL 0 // Front-Left
#define MOTOR_BL 1 // Back-Left
#define MOTOR_FR 2 // Front-Right
#define MOTOR_BR 3 // Back-Right

// Nuvoton reset pin — must be pulsed LOW then HIGH before I2C motor
// commands will be accepted (mirrors Nuvoton::begin() in Nuvoton.cpp)
#define WSNV_PIN_RESET 33

// ─── CAMERA PIN DEFINITIONS
// ─────────────────────────────────────────────────── Matches
// Wheelson-Library/src/Pins.hpp (all hardware revisions)
#define PWDN_GPIO_NUM -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 4
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// ─── GLOBALS ─────────────────────────────────────────────────────────────────

WebServer server(80);

const char *navStateName(NavState s) {
  switch (s) {
  case NAV_HOLD:
    return "HOLD";
  case NAV_CRUISE:
    return "CRUISE";
  default:
    return "UNKNOWN";
  }
}

// ─── NUVOTON INIT ────────────────────────────────────────────────────────────
/**
 * Mirrors Nuvoton::begin() from the Wheelson library.
 * The Nuvoton co-processor ignores all I2C commands until it has
 * received this reset pulse and had 500 ms to boot.
 */
void nuvotonBegin() {
  pinMode(WSNV_PIN_RESET, OUTPUT);
  digitalWrite(WSNV_PIN_RESET, LOW);
  delay(50);
  digitalWrite(WSNV_PIN_RESET, HIGH);
  delay(500); // wait for Nuvoton to come online
  Serial.println("[NUVOTON] Reset complete");
}

// ─── FRAME ANALYSIS TIMING ──────────────────────────────────────────────────
unsigned long lastAnalysisTime = 0;
unsigned long lastSuccessfulAnalysisTime = 0;
bool hasAnalysis = false;
const int ANALYSIS_INTERVAL_MS = 150;

// ─── JSON HELPERS ────────────────────────────────────────────────────────────
// Lightweight replacements for ArduinoJson — no external library needed.

/**
 * Escape a string for safe inclusion in a JSON value.
 * Handles quotes, backslashes, and control characters.
 */
String jsonEscape(const String &s) {
  String result;
  result.reserve(s.length() + 4);
  for (unsigned int i = 0; i < s.length(); i++) {
    char c = s.charAt(i);
    if (c == '"') result += "\\\"";
    else if (c == '\\') result += "\\\\";
    else if (c == '\n') result += "\\n";
    else if (c == '\r') result += "\\r";
    else result += c;
  }
  return result;
}

/**
 * Extract a string value from a flat JSON object.
 * e.g. extractJsonString(body, "action") returns "forward"
 */
String extractJsonString(const String &json, const String &key) {
  String search = "\"" + key + "\"";
  int keyIdx = json.indexOf(search);
  if (keyIdx < 0)
    return "";
  int colon = json.indexOf(':', keyIdx + search.length());
  if (colon < 0)
    return "";
  int q1 = json.indexOf('"', colon + 1);
  if (q1 < 0)
    return "";
  int q2 = json.indexOf('"', q1 + 1);
  if (q2 < 0)
    return "";
  return json.substring(q1 + 1, q2);
}

/**
 * Extract an integer value from a flat JSON object.
 * e.g. extractJsonInt(body, "duration_ms") returns 500
 */
int extractJsonInt(const String &json, const String &key) {
  String search = "\"" + key + "\"";
  int keyIdx = json.indexOf(search);
  if (keyIdx < 0)
    return 0;
  int colon = json.indexOf(':', keyIdx + search.length());
  if (colon < 0)
    return 0;
  int start = colon + 1;
  while (start < (int)json.length() && json[start] == ' ')
    start++;
  return json.substring(start).toInt();
}

bool consumeSafetyLatch(String &reason) {
  bool fired = safetyStopLatched;
  reason = safetyStopReason;
  if (fired) {
    safetyStopLatched = false;
    safetyStopReason = "";
  }
  return fired;
}

// ─── MOTOR CONTROL ───────────────────────────────────────────────────────────
//
// All driving goes through slew-limited wheel targets: commands set TARGET
// intensities, and a dedicated task ramps the output toward them so starts,
// stops, and turns are eased instead of bang-bang.

#define MOTOR_COUNT 4
int8_t motorTarget[MOTOR_COUNT] = {0, 0, 0, 0};
int8_t motorCurrent[MOTOR_COUNT] = {0, 0, 0, 0};
MotionProfile motorProfiles[MOTOR_COUNT];
SemaphoreHandle_t motorMutex;
SemaphoreHandle_t i2cMutex;
bool motorLeaseArmed = false;
unsigned long motorLeaseDeadline = 0;
unsigned long motorTimedDeadline = 0;
bool motorTimed = false;
bool motorVisionReady = false;
unsigned long motorVisionDeadline = 0;
int motorEvent = 0; // 1 = watchdog, 2 = timed completion, 3 = stale vision.
String authoritySession;
unsigned long authoritySequence = 0;
String motionCommandId;

void setMotor(uint8_t id, int8_t intensity) {
  xSemaphoreTake(i2cMutex, portMAX_DELAY);
  Wire.beginTransmission(NUVOTON_ADDR);
  Wire.write(MOTOR_SET_BYTE);
  Wire.write(id);
  Wire.write(static_cast<uint8_t>(intensity));
  Wire.endTransmission();
  xSemaphoreGive(i2cMutex);
}

void zeroMotorsLocked() {
  motorLeaseArmed = motorTimed = false;
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i) {
    motorTarget[i] = motorCurrent[i] = 0;
    motorProfiles[i].reset();
    setMotor(i, 0);
  }
}

void emergencyStopMotors() {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  zeroMotorsLocked();
  motorEvent = 0;
  xSemaphoreGive(motorMutex);
}

void motorControlTask(void *) {
  TickType_t wake = xTaskGetTickCount();
  unsigned long previous = millis();
  for (;;) {
    xSemaphoreTake(motorMutex, portMAX_DELAY);
    unsigned long now = millis();
    float dt = (now - previous) / 1000.0f;
    previous = now;
    if (motorLeaseArmed && motorTarget[MOTOR_FL] > 0 && motorTarget[MOTOR_FR] > 0 &&
        (!motorVisionReady || (long)(now - motorVisionDeadline) >= 0)) {
      zeroMotorsLocked();
      motorEvent = 3;
    } else if (motorLeaseArmed && (long)(now - motorLeaseDeadline) >= 0) {
      zeroMotorsLocked();
      motorEvent = 1;
    } else if (motorTimed && (long)(now - motorTimedDeadline) >= 0) {
      for (int i = 0; i < MOTOR_COUNT; ++i) motorTarget[i] = 0;
      motorLeaseArmed = motorTimed = false;
      motorEvent = 2;
    }
    for (uint8_t i = 0; i < MOTOR_COUNT; ++i) {
      int8_t next = motorProfiles[i].step(motorTarget[i], dt);
      if (next != motorCurrent[i]) {
        motorCurrent[i] = next;
        setMotor(i, next);
      }
    }
    xSemaphoreGive(motorMutex);
    // Do not run back-to-back catch-up ticks after slow I2C work.
    if (xTaskGetTickCount() - wake >= pdMS_TO_TICKS(20)) wake = xTaskGetTickCount();
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(20));
  }
}

void syncMotorEvent() {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  int event = motorEvent;
  motorEvent = 0;
  xSemaphoreGive(motorMutex);
  if (!event) return;
  currentMoveAction = commandedAction = "stop";
  isMovingIndefinitely = false;
  navState = NAV_HOLD;
  motionCommandId = "";
  if (event == 1 || event == 3) {
    safetyStopLatched = true;
    safetyStopReason = event == 1 ? "lease_timeout" : "perception_timeout";
  }
}

void setWheelTargets(int8_t fl, int8_t bl, int8_t fr, int8_t br) {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  if ((motorEvent == 1 || motorEvent == 3) && (fl || bl || fr || br)) {
    xSemaphoreGive(motorMutex);
    return; // Watchdog must be acknowledged before any restart.
  }
  motorEvent = 0;
  motorTarget[MOTOR_FL] = fl;
  motorTarget[MOTOR_BL] = bl;
  motorTarget[MOTOR_FR] = fr;
  motorTarget[MOTOR_BR] = br;
  motorLeaseArmed = fl || bl || fr || br;
  motorLeaseDeadline = millis() + COMMAND_LEASE_TIMEOUT_MS;
  motorTimed = false;
  xSemaphoreGive(motorMutex);
}

bool renewMotorLease() {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  bool valid = motorLeaseArmed && (long)(millis() - motorLeaseDeadline) < 0;
  if (valid) motorLeaseDeadline = millis() + COMMAND_LEASE_TIMEOUT_MS;
  xSemaphoreGive(motorMutex);
  return valid;
}

void setTimedDeadline(int duration) {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  motorTimed = true;
  motorTimedDeadline = millis() + duration;
  xSemaphoreGive(motorMutex);
}

bool targetsForward() {
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  bool forward = motorTarget[MOTOR_FL] > 0 && motorTarget[MOTOR_FR] > 0;
  xSemaphoreGive(motorMutex);
  return forward;
}

void driveDifferential(int8_t left, int8_t right) {
  setWheelTargets(left, left, right, right);
}

void setHeadlight(uint8_t brightness) {
  xSemaphoreTake(i2cMutex, portMAX_DELAY);
  Wire.beginTransmission(NUVOTON_ADDR);
  Wire.write(HEADLIGHT_SET_BYTE);
  Wire.write(brightness);
  Wire.endTransmission();
  xSemaphoreGive(i2cMutex);
}

void stopMotors() { setWheelTargets(0, 0, 0, 0); }

// Drive all four wheels forward
void driveForward() {
  int8_t forwardSpeed = (int8_t)(currentSpeed * FORWARD_SPEED_SCALE);
  setWheelTargets(forwardSpeed, forwardSpeed, forwardSpeed, forwardSpeed);
}

// Drive all four wheels backward
void driveBackward() {
  setWheelTargets(-currentSpeed, -currentSpeed, -currentSpeed, -currentSpeed);
}

// Left wheels backward, right wheels forward → pivot left
void turnLeft() {
  setWheelTargets(-currentSpeed, -currentSpeed, currentSpeed, currentSpeed);
}

// Left wheels forward, right wheels backward → pivot right
void turnRight() {
  setWheelTargets(currentSpeed, currentSpeed, -currentSpeed, -currentSpeed);
}

// ─── DISTANCE SENSOR ─────────────────────────────────────────────────────────

long lastEchoDuration = 0;

/**
 * Trigger HC-SR04 pulse and measure echo time.
 * Returns distance in centimetres, or 999.0 on timeout (clear path).
 */
float readDistanceCM() {
  if (!ULTRASONIC_ENABLED) {
    lastEchoDuration = 0;
    return 999.0f;
  }

  for (int i = 0; i < 3; i++) {
    noInterrupts();
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);
    interrupts();

    // 35 ms timeout → max ~600 cm
    lastEchoDuration = pulseIn(ECHO_PIN, HIGH, 35000UL);
    if (lastEchoDuration > 0) {
      break; // Success
    }
    delay(10); // Short delay before pinging again to let original echoes
               // dissipate
  }

  if (lastEchoDuration == 0) {
    return 999.0f;
  }

  return lastEchoDuration * 0.0343f / 2.0f;
}

// ─── HYBRID EDGE AI (VISUAL OBSTACLE DETECTION) ──────────────────────────────

/**
 * Perform a fast pixel-level analysis of a QQVGA (160x120) RGB888 frame buffer.
 * Estimates floor obstacles by comparing lookahead color to safe floor color.
 * Uses an EMA on ratios, a latched decision with hysteresis, and a rolling
 * floor-color reference that only updates while the frame looks clear.
 */
void analyzeFrame(uint8_t *rgb888_pixels, int width, int height) {
  long r_sum = 0, g_sum = 0, b_sum = 0;
  int safe_count = 0;

  // 1. Sample the "safe floor" (rows height-40 to height-15, center 40 columns)
  // We avoid the absolute bottom to prevent sampling the Wheelson's own
  // nose/wires
  for (int y = height - 40; y < height - 15; y++) {
    for (int x = width / 2 - 20; x < width / 2 + 20; x++) {
      size_t offset = (y * width + x) * 3;
      // Preserve the existing RGB565-scale tuning while decoding RGB888.
      r_sum += rgb888_pixels[offset] >> 3;
      g_sum += rgb888_pixels[offset + 1] >> 2;
      b_sum += rgb888_pixels[offset + 2] >> 3;
      safe_count++;
    }
  }

  if (safe_count == 0)
    return;
  int raw_r = r_sum / safe_count;
  int raw_g = g_sum / safe_count;
  int raw_b = b_sum / safe_count;

  // Rolling reference: blend toward the new sample only while unobstructed.
  if (!telemetryObstructed) {
    if (refFloorR < 0) {
      refFloorR = raw_r;
      refFloorG = raw_g;
      refFloorB = raw_b;
    } else {
      refFloorR += (raw_r - refFloorR) * FLOOR_REF_ALPHA_NUM / FLOOR_REF_ALPHA_DEN;
      refFloorG += (raw_g - refFloorG) * FLOOR_REF_ALPHA_NUM / FLOOR_REF_ALPHA_DEN;
      refFloorB += (raw_b - refFloorB) * FLOOR_REF_ALPHA_NUM / FLOOR_REF_ALPHA_DEN;
    }
  }
  int ref_r = refFloorR;
  int ref_g = refFloorG;
  int ref_b = refFloorB;

  // 2. Scan the "lookahead" area (rows height/2 - 10 to height-40)
  int obstacle_pixels = 0;
  int total_lookahead = 0;
  int obstacle_left = 0;
  int obstacle_right = 0;
  int total_left = 0;
  int total_right = 0;
  long total_luma = 0;
  long look_r_sum = 0, look_g_sum = 0, look_b_sum = 0;

  for (int y = (height / 2) - 10; y < height - 40; y += 2) {
    for (int x = 40; x < width - 40;
         x += 2) { // 40-pixel inset to skip dark vignetting
      size_t offset = (y * width + x) * 3;
      int r = rgb888_pixels[offset] >> 3;
      int g = rgb888_pixels[offset + 1] >> 2;
      int b = rgb888_pixels[offset + 2] >> 3;

      look_r_sum += r;
      look_g_sum += g;
      look_b_sum += b;
      total_luma += r + (g / 2) + b; // rough luma, out of ~94

      // Euclidean distance in RGB space to the reference floor:
      int dr = r - ref_r;
      int dg = g - ref_g;
      int db = b - ref_b;
      // Scale G since it has 6 bits vs 5 bits
      int dist_sq = (dr * dr) + ((dg / 2) * (dg / 2)) + (db * db);

      // Threshold for "different enough to be an obstacle"
      // 200 is a balanced value for OV2640 sensor noise vs real edges
      bool is_obstacle = false;
      if (dist_sq > 200) {
        obstacle_pixels++;
        is_obstacle = true;
      }
      if (x < width / 2) {
        total_left++;
        if (is_obstacle)
          obstacle_left++;
      } else {
        total_right++;
        if (is_obstacle)
          obstacle_right++;
      }
      total_lookahead++;
    }
  }

  if (total_lookahead == 0)
    return;

  // EMA smoothing of raw ratios before any decision uses them.
  float rawRatio = (float)obstacle_pixels / total_lookahead;
  float rawLeft = total_left > 0 ? ((float)obstacle_left / total_left) : 0.0f;
  float rawRight = total_right > 0 ? ((float)obstacle_right / total_right) : 0.0f;
  if (!emaSeeded) {
    emaObstacleLeftRatio = rawLeft;
    emaObstacleRightRatio = rawRight;
    emaSeeded = true;
  } else {
    emaObstacleLeftRatio += OBSTACLE_EMA_ALPHA * (rawLeft - emaObstacleLeftRatio);
    emaObstacleRightRatio += OBSTACLE_EMA_ALPHA * (rawRight - emaObstacleRightRatio);
  }
  telemetryObstacleLeftRatio = emaObstacleLeftRatio;
  telemetryObstacleRightRatio = emaObstacleRightRatio;

  // Latched obstruction with hysteresis: enter high, exit low.
  float exitRatio = obstacleEnterRatio * OBSTACLE_EXIT_RATIO_FACTOR;
  if (telemetryObstructed) {
    if (rawRatio < exitRatio)
      telemetryObstructed = false;
  } else {
    if (rawRatio > obstacleEnterRatio)
      telemetryObstructed = true;
  }

  int avg_luma = total_luma / total_lookahead;
  if (avg_luma < 15)
    telemetryBrightness = "Dark";
  else if (avg_luma > 70)
    telemetryBrightness = "Bright";
  else
    telemetryBrightness = "Normal";

  int dom_r = (look_r_sum / total_lookahead) * 8;
  int dom_g = (look_g_sum / total_lookahead) * 4;
  int dom_b = (look_b_sum / total_lookahead) * 8;

  char hex[8];
  sprintf(hex, "#%02X%02X%02X", dom_r, dom_g, dom_b);
  telemetryColorHex = String(hex);
}

// ─── CAMERA ──────────────────────────────────────────────────────────────────

bool initCamera() {
  // The ESP32 camera config gained fields in recent Arduino-ESP32 cores.
  // Zero-initialize it so those fields never carry stack garbage.
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
#else
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
#endif
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  // Wheelson's camera module is unreliable above 10 MHz; a lower clock avoids
  // DMA frame-buffer overruns on this wiring revision.
  config.xclk_freq_hz = 10000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QQVGA; // 160×120 — small & fast
  config.jpeg_quality = 10;            // 10 = high quality (lower = better)
  // The capture DMA continues while frame analysis is running. Use two PSRAM
  // buffers and keep the newest completed image so it cannot overrun a single
  // held buffer (cam_hal: FB-OVF).
  config.fb_count = 2;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  config.fb_location = CAMERA_FB_IN_PSRAM; // requires Tools → PSRAM → Enabled
  config.grab_mode = CAMERA_GRAB_LATEST;
#endif

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAMERA] Init failed: 0x%x\n", err);
    return false;
  }

  // Improve image saturation & disable special effects
  sensor_t *s = esp_camera_sensor_get();
  s->set_special_effect(s, 0);
  s->set_saturation(s, 2);
  // Camera is physically inverted: correct it once for safety, VLM and UI.
  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);

  Serial.println("[CAMERA] Ready (160×120 JPEG)");
  return true;
}

// ─── HTTP HANDLERS ───────────────────────────────────────────────────────────

/**
 * GET /camera
 * Returns raw JPEG bytes.
 * X-Distance-CM header carries the current sensor reading.
 */
void handleCamera() {
  float dist = readDistanceCM();

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "text/plain", "Camera capture failed");
    return;
  }

  // Telemetry headers use cached values from the periodic analysis in loop()
  server.sendHeader("X-Distance-CM", String(dist, 1));
  server.sendHeader("X-Analysis-Age-MS", String(hasAnalysis ? millis() - lastSuccessfulAnalysisTime : 999999UL));
  server.sendHeader("X-Frame-Orientation", "upright");
  xSemaphoreTake(motorMutex, portMAX_DELAY);
  int outputLeft = motorCurrent[MOTOR_FL], outputRight = motorCurrent[MOTOR_FR];
  int targetLeft = motorTarget[MOTOR_FL], targetRight = motorTarget[MOTOR_FR];
  xSemaphoreGive(motorMutex);
  server.sendHeader("X-Wheel-Output-Left", String(outputLeft));
  server.sendHeader("X-Wheel-Output-Right", String(outputRight));
  server.sendHeader("X-Wheel-Target-Left", String(targetLeft));
  server.sendHeader("X-Wheel-Target-Right", String(targetRight));
  server.sendHeader("X-Raw-Pulse-US", String(lastEchoDuration));
  server.sendHeader("X-Visual-Obstacle",
                    telemetryObstructed ? "true" : "false");
  server.sendHeader("X-Obstacle-Left-Ratio", String(telemetryObstacleLeftRatio, 3));
  server.sendHeader("X-Obstacle-Right-Ratio", String(telemetryObstacleRightRatio, 3));
  server.sendHeader("X-Nav-State", String(navStateName(navState)));
  server.sendHeader("X-Brightness", telemetryBrightness);
  server.sendHeader("X-Dominant-Color", telemetryColorHex);
  server.sendHeader("X-Active-Command-Id", activeCommandId);
  server.sendHeader("X-Active-Command-Source", activeCommandSource);
  server.sendHeader("X-Active-Command-Mode", activeCommandMode);
  server.sendHeader("X-Safety-Latched", safetyStopLatched ? "true" : "false");
  server.sendHeader("X-Safety-Reason", safetyStopReason);
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Cache-Control", "no-cache, no-store");

  server.setContentLength(fb->len);
  server.send(200, "image/jpeg", "");
  server.client().write(reinterpret_cast<const char *>(fb->buf), fb->len);

  esp_camera_fb_return(fb);
}

/**
 * GET /status
 * Returns: { "distance_cm": <float>, "ip": "<string>" }
 */
void handleStatus() {
  float dist = readDistanceCM();

  String body = "{\"distance_cm\":\"" + String(dist, 1) + "\",\"ip\":\"" +
                jsonEscape(WiFi.localIP().toString()) + "\",\"nav_state\":\"" +
                String(navStateName(navState)) + "\",\"command_id\":\"" +
                jsonEscape(activeCommandId) + "\",\"source\":\"" + jsonEscape(activeCommandSource) +
                "\",\"mode\":\"" + jsonEscape(activeCommandMode) + "\",\"safety_latched\":" +
                String(safetyStopLatched ? "true" : "false") +
                ",\"safety_reason\":\"" + jsonEscape(safetyStopReason) + "\"}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", body);
}

/**
 * POST /move
 * Body: { "action": "forward|backward|left|right|stop", "duration_ms": 500 }
 * Applies hard-stop safety override before driving.
 * Returns: { "ok": true, "action": <actual>, "duration_ms": <int>,
 * "distance_cm": <float> }
 */
void handleMove() {
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "JSON body required");
    return;
  }

  String body = server.arg("plain");
  String command = extractJsonString(body, "command");
  String action = extractJsonString(body, "action");
  int duration_ms = extractJsonInt(body, "duration_ms");
  String commandId = extractJsonString(body, "command_id");
  syncMotorEvent();
  String session = extractJsonString(body, "session_id");
  int sequenceInput = extractJsonInt(body, "sequence");
  unsigned long seq = (unsigned long)sequenceInput;
  if (sequenceInput <= 0) {
    server.send(400, "application/json", "{\"ok\":false}");
    return;
  }
  if (command == "claim" && session.length() == 32 &&
      (session != authoritySession || seq > authoritySequence)) {
    emergencyStopMotors();
    authoritySession = session;
    authoritySequence = seq;
    motionCommandId = "";
    currentMoveAction = commandedAction = "stop";
    isMovingIndefinitely = false;
    navState = NAV_HOLD;
    server.send(200, "application/json", "{\"ok\":true,\"protocol\":2}");
    return;
  }
  if (session != authoritySession || session.length() == 0 || seq <= authoritySequence) {
    server.send(409, "application/json", "{\"ok\":false,\"busy\":true}");
    return;
  }
  authoritySequence = seq;
  if (command == "renew") {
    bool valid = extractJsonString(body, "target_id") == motionCommandId &&
                 motionCommandId.length() > 0 && !safetyStopLatched && renewMotorLease();
    server.send(valid ? 200 : 409, "application/json",
                valid ? "{\"ok\":true}" : "{\"ok\":false,\"busy\":true}");
    return;
  }

  String source = extractJsonString(body, "source");
  String mode = extractJsonString(body, "mode");

  if (commandId.length() > 0)
    activeCommandId = commandId;
  if (source.length() > 0)
    activeCommandSource = source;
  if (mode.length() > 0)
    activeCommandMode = mode;

  String visSafety = extractJsonString(body, "visual_safety");
  if (visSafety == "false") {
    visualObstacleSafetyEnabled = false;
  } else if (visSafety == "true") {
    visualObstacleSafetyEnabled = true;
  }

  // Optional runtime tuning of the obstruction enter threshold
  // (thousandths, e.g. 400 = 0.40). Applies to any /move command.
  int thrMille = extractJsonInt(body, "obstacle_threshold");
  if (thrMille > 0 && thrMille <= 1000) {
    obstacleEnterRatio = thrMille / 1000.0f;
  }

  String safetyReason = safetyStopReason;
  bool safety_fired = safetyStopLatched;
  if (safetyReason.length() == 0) {
    safetyReason = "none";
  }

  if (command == "set_speed") {
    String level = extractJsonString(body, "level");
    if (level == "slow")
      currentSpeed = SPEED_SLOW;
    else if (level == "fast")
      currentSpeed = SPEED_FAST;
    else
      currentSpeed = SPEED_MEDIUM; // medium or default

    int forwardSpeed = (int)(currentSpeed * FORWARD_SPEED_SCALE);
    Serial.printf("[SPEED] level=%s base=%d forward=%d\n", level.c_str(),
                  (int)currentSpeed, forwardSpeed);

    String boolStr = safety_fired ? "true" : "false";
    String resp = "{\"ok\":true,\"command\":\"set_speed\",\"level\":\"" + jsonEscape(level) +
                  "\",\"base_speed\":" + String((int)currentSpeed) +
                  ",\"forward_speed\":" + String(forwardSpeed) +
                  ",\"safety\":" + boolStr + ",\"safety_reason\":\"" +
                  jsonEscape(safetyReason) + "\",\"command_id\":\"" + jsonEscape(activeCommandId) +
                  "\",\"source\":\"" + jsonEscape(activeCommandSource) +
                  "\",\"mode\":\"" + jsonEscape(activeCommandMode) + "\"}";
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", resp);
    return;
  } else if (command == "set_light") {
    String level = extractJsonString(body, "level");
    uint8_t brightness = level.toInt();
    setHeadlight(brightness);

    String boolStr = safety_fired ? "true" : "false";
    String resp = "{\"ok\":true,\"command\":\"set_light\",\"level\":\"" + jsonEscape(level) +
                  "\",\"safety\":" + boolStr + ",\"safety_reason\":\"" +
                  jsonEscape(safetyReason) + "\",\"command_id\":\"" + jsonEscape(activeCommandId) +
                  "\",\"source\":\"" + jsonEscape(activeCommandSource) +
                  "\",\"mode\":\"" + jsonEscape(activeCommandMode) + "\"}";
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", resp);
    return;
  } else if (command == "stop") {
    String consumedSafetyReason;
    safety_fired = consumeSafetyLatch(consumedSafetyReason);
    if (consumedSafetyReason.length() > 0) {
      safetyReason = consumedSafetyReason;
    } else if (safetyReason.length() == 0) {
      safetyReason = "none";
    }

    if (source == "failsafe" || source == "shutdown" || source == "lease_rejected")
      emergencyStopMotors();
    else
      stopMotors();
    motionCommandId = "";
    action = "stop";
    commandedAction = "stop";
    currentMoveAction = "stop";
    isMovingIndefinitely = false;
    navState = NAV_HOLD;
    String boolStr = safety_fired ? "true" : "false";
    String resp = "{\"ok\":true,\"action\":\"stop\",\"duration_ms\":0,\"safety\":" +
                  boolStr + ",\"safety_reason\":\"" + jsonEscape(safetyReason) +
                  "\",\"busy\":false,\"command_id\":\"" + jsonEscape(activeCommandId) +
                  "\",\"source\":\"" + jsonEscape(activeCommandSource) +
                  "\",\"mode\":\"" + jsonEscape(activeCommandMode) +
                  "\",\"nav_state\":\"" + String(navStateName(navState)) + "\"}";
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", resp);
    return;
  }

  String consumedSafetyReason;
  safety_fired = consumeSafetyLatch(consumedSafetyReason);
  if (consumedSafetyReason.length() > 0) {
    safetyReason = consumedSafetyReason;
  } else if (safetyReason.length() == 0) {
    safetyReason = "none";
  }

  bool requestedForward = (action == "forward") ||
      (command == "move_indefinitely" && extractJsonString(body, "direction") == "forward") ||
      (command == "drive" && extractJsonInt(body, "left") > 0 && extractJsonInt(body, "right") > 0);
  if (safety_fired || (visualObstacleSafetyEnabled && telemetryObstructed && requestedForward)) {
    emergencyStopMotors();
    motionCommandId = "";
    currentMoveAction = commandedAction = "stop";
    isMovingIndefinitely = false;
    navState = NAV_HOLD;
    server.send(200, "application/json", "{\"ok\":false,\"action\":\"stop\",\"safety\":true}");
    return;
  }
  motionCommandId = commandId;
  float dist = readDistanceCM();

  if (command == "move_indefinitely") {
    String direction = extractJsonString(body, "direction");
    action = direction;
    duration_ms = 0;
    isMovingIndefinitely = (action != "stop");
    commandedAction = action;
  } else if (command == "drive") {
    // Differential drive: signed side intensities, lease-renewed like
    // move_indefinitely. The motor task eases the transition.
    int leftI = constrain(extractJsonInt(body, "left"), -127, 127);
    int rightI = constrain(extractJsonInt(body, "right"), -127, 127);

    consumeSafetyLatch(consumedSafetyReason);
    driveDifferential((int8_t)leftI, (int8_t)rightI);
    bool moving = (leftI != 0 || rightI != 0);
    currentMoveAction = moving ? "drive" : "stop";
    commandedAction = currentMoveAction;
    isMovingIndefinitely = moving;
    lastDistanceCheckTime = millis();
    navState = moving ? NAV_CRUISE : NAV_HOLD;

    String dResp = "{\"ok\":true,\"action\":\"drive\",\"left\":" + String(leftI) +
                   ",\"right\":" + String(rightI) + ",\"duration_ms\":0,\"safety\":" +
                   String(safety_fired ? "true" : "false") + ",\"safety_reason\":\"" +
                   jsonEscape(safetyReason) + "\",\"busy\":false,\"command_id\":\"" +
                   jsonEscape(activeCommandId) + "\",\"source\":\"" +
                   jsonEscape(activeCommandSource) + "\",\"mode\":\"" +
                   jsonEscape(activeCommandMode) + "\",\"nav_state\":\"" +
                   String(navStateName(navState)) + "\"}";
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", dResp);
    return;
  } else {
    // Legacy parsing
    if (action.length() == 0)
      action = "stop";
    if (duration_ms <= 0)
      duration_ms = 500;
    isMovingIndefinitely = false;
    commandedAction = action;
  }

  if (!(action == "forward" || action == "backward" || action == "left" ||
        action == "right" || action == "stop")) {
    action = "stop";
    duration_ms = 0;
    isMovingIndefinitely = false;
    commandedAction = "stop";
  }

  // Update state variables
  currentMoveAction = action;
  lastDistanceCheckTime = millis();
  navState = (action == "stop") ? NAV_HOLD : NAV_CRUISE;

  duration_ms = constrain(duration_ms, 0, 3000);

  // Execute drive logic
  if (action == "forward")
    driveForward();
  else if (action == "backward")
    driveBackward();
  else if (action == "left")
    turnLeft();
  else if (action == "right")
    turnRight();
  else
    stopMotors();

  if (!isMovingIndefinitely && action != "stop") setTimedDeadline(duration_ms);

  String boolStr = safety_fired ? "true" : "false";
  String resp = "{\"ok\":true,\"action\":\"" + jsonEscape(action) +
                "\",\"duration_ms\":" + duration_ms + ",\"distance_cm\":\"" +
                String(dist, 1) + "\",\"safety\":" + boolStr +
                ",\"safety_reason\":\"" + jsonEscape(safetyReason) +
                "\",\"busy\":false,\"command_id\":\"" + jsonEscape(activeCommandId) +
                "\",\"source\":\"" + jsonEscape(activeCommandSource) +
                "\",\"mode\":\"" + jsonEscape(activeCommandMode) + "\",\"nav_state\":\"" +
                String(navStateName(navState)) + "\"}";

  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", resp);
}

// Handle CORS pre-flight for browsers / curl
void handleOptions() {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  server.send(204);
}

// ─── SETUP & LOOP ────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n╔══════════════════════════════════╗");
  Serial.println("║  Wheelson Autonomous Explorer    ║");
  Serial.println("╚══════════════════════════════════╝");

  // Distance sensor (disabled by default; kept for compatibility)
  if (ULTRASONIC_ENABLED) {
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    digitalWrite(TRIG_PIN, LOW);
  }

  // Nuvoton co-processor reset (MUST happen before Wire.begin + motor cmds)
  nuvotonBegin();

  // I2C → Nuvoton motor controller
  Wire.begin(I2C_SDA, I2C_SCL);
  motorMutex = xSemaphoreCreateMutex();
  i2cMutex = xSemaphoreCreateMutex();
  if (!motorMutex || !i2cMutex) { Serial.println("Motor mutex allocation failed"); while (true) delay(1000); }
  Wire.setTimeOut(10);
  emergencyStopMotors();
  if (xTaskCreate(motorControlTask, "motor-control", 4096, nullptr, 3, nullptr) != pdPASS) {
    Serial.println("Motor task allocation failed"); while (true) delay(1000);
  }
  Serial.println("[MOTORS] I2C ready");

  // Camera
  if (!initCamera()) {
    Serial.println("[ERROR] Camera init failed — halting");
    while (true)
      delay(1000);
  }

  // Allocate PSRAM buffer for RGB888 decoding.
  visual_rgb_buf = (uint8_t *)ps_malloc(160 * 120 * 3);
  if (!visual_rgb_buf) {
    Serial.println("[ERROR] Failed to allocate visual_rgb_buf in PSRAM");
  }

  // Wi-Fi
  Serial.printf("[WIFI] Connecting to '%s'", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.printf("\n[WIFI] Connected! IP: %s\n",
                WiFi.localIP().toString().c_str());

  // HTTP routes
  server.on("/camera", HTTP_GET, handleCamera);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/move", HTTP_POST, handleMove);
  server.on("/camera", HTTP_OPTIONS, handleOptions);
  server.on("/move", HTTP_OPTIONS, handleOptions);
  server.begin();

  Serial.println("[HTTP] Server started on port 80");
  Serial.println("[READY] Endpoints:");
  Serial.printf("  GET  http://%s/camera\n", WiFi.localIP().toString().c_str());
  Serial.printf("  GET  http://%s/status\n", WiFi.localIP().toString().c_str());
  Serial.printf("  POST http://%s/move\n", WiFi.localIP().toString().c_str());
}

void loop() {
  server.handleClient();
  syncMotorEvent();

  unsigned long now = millis();

  // Periodic frame analysis — runs unconditionally so telemetry is always fresh
  // for both the safety check and /camera HTTP responses.
  if (now - lastAnalysisTime >= ANALYSIS_INTERVAL_MS) {
    lastAnalysisTime = now;

    if (visual_rgb_buf) {
      camera_fb_t *fb = esp_camera_fb_get();
      if (fb) {
        if (fmt2rgb888(fb->buf, fb->len, fb->format, visual_rgb_buf)) {
          analyzeFrame(visual_rgb_buf, fb->width, fb->height);
          lastSuccessfulAnalysisTime = millis();
          hasAnalysis = true;
          xSemaphoreTake(motorMutex, portMAX_DELAY);
          motorVisionReady = true;
          motorVisionDeadline = millis() + 800;
          xSemaphoreGive(motorMutex);
        }
        esp_camera_fb_return(fb);
      }
    }

    // Reactive hard-stop: if cruising forward (plain or arc) and vision
    // detects an obstacle ahead.
    bool drivingForward =
        targetsForward();
    if (visualObstacleSafetyEnabled && navState == NAV_CRUISE && drivingForward &&
        telemetryObstructed) {
      emergencyStopMotors();
      motionCommandId = "";
      currentMoveAction = "stop";
      commandedAction = "stop";
      isMovingIndefinitely = false;
      navState = NAV_HOLD;
      safetyStopLatched = true;
      safetyStopReason = "visual_obstacle";
      Serial.printf(
          "[SAFETY] HARD_STOP reason=visual_obstacle sideL=%.2f sideR=%.2f\n",
          telemetryObstacleLeftRatio, telemetryObstacleRightRatio);
    }
  }

}
