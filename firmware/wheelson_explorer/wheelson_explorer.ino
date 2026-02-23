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

// ─── USER CONFIGURATION ──────────────────────────────────────────────────────

#define WIFI_SSID "WIFI_SSID"
#define WIFI_PASSWORD "WIFI_PASSWORD"

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
unsigned long targetStopTime = 0;
unsigned long lastDistanceCheckTime = 0;
unsigned long lastLeaseCommandTime = 0;
const int DISTANCE_CHECK_INTERVAL_MS =
    150; // Increased to 150ms for reliable echoes
const unsigned long COMMAND_LEASE_TIMEOUT_MS = 1500;

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

// Last accepted command metadata (from middleware authority)
String activeCommandId = "boot";
String activeCommandSource = "bootstrap";
String activeCommandMode = "idle";

// Sticky one-shot safety latch consumed by /move responses
bool safetyStopLatched = false;
String safetyStopReason = "";

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

// ─── JSON HELPERS ────────────────────────────────────────────────────────────
// Lightweight replacements for ArduinoJson — no external library needed.

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

/**
 * Send a signed-intensity command to one motor via I2C.
 * Matches the byte protocol in WheelsonMotor::setMotor().
 */
void setMotor(uint8_t id, int8_t intensity) {
  Wire.beginTransmission(NUVOTON_ADDR);
  Wire.write(MOTOR_SET_BYTE);
  Wire.write(id);
  Wire.write(static_cast<uint8_t>(intensity));
  Wire.endTransmission();
}

/**
 * Send a brightness command to the headlights via I2C.
 * Matches the byte protocol in WheelsonLED::setHeadlight().
 */
void setHeadlight(uint8_t brightness) {
  Wire.beginTransmission(NUVOTON_ADDR);
  Wire.write(HEADLIGHT_SET_BYTE);
  Wire.write(brightness);
  Wire.endTransmission();
}

void stopMotors() {
  for (uint8_t i = 0; i < 4; i++)
    setMotor(i, 0);
}

// Drive all four wheels forward
void driveForward() {
  int8_t forwardSpeed = (int8_t)(currentSpeed * FORWARD_SPEED_SCALE);
  setMotor(MOTOR_FL, forwardSpeed);
  setMotor(MOTOR_BL, forwardSpeed);
  setMotor(MOTOR_FR, forwardSpeed);
  setMotor(MOTOR_BR, forwardSpeed);
}

// Drive all four wheels backward
void driveBackward() {
  setMotor(MOTOR_FL, -currentSpeed);
  setMotor(MOTOR_BL, -currentSpeed);
  setMotor(MOTOR_FR, -currentSpeed);
  setMotor(MOTOR_BR, -currentSpeed);
}

// Left wheels backward, right wheels forward → pivot left
void turnLeft() {
  setMotor(MOTOR_FL, -currentSpeed);
  setMotor(MOTOR_BL, -currentSpeed);
  setMotor(MOTOR_FR, currentSpeed);
  setMotor(MOTOR_BR, currentSpeed);
}

// Left wheels forward, right wheels backward → pivot right
void turnRight() {
  setMotor(MOTOR_FL, currentSpeed);
  setMotor(MOTOR_BL, currentSpeed);
  setMotor(MOTOR_FR, -currentSpeed);
  setMotor(MOTOR_BR, -currentSpeed);
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
 * Perform a fast pixel-level analysis of a QQVGA (160x120) RGB565 frame buffer.
 * Estimates floor obstacles by comparing lookahead color to safe floor color.
 */
void analyzeFrame(uint8_t *rgb565_pixels, int width, int height) {
  uint16_t *pixels = (uint16_t *)rgb565_pixels;
  long r_sum = 0, g_sum = 0, b_sum = 0;
  int safe_count = 0;

  // 1. Sample the "safe floor" (rows height-40 to height-15, center 40 columns)
  // We avoid the absolute bottom to prevent sampling the Wheelson's own
  // nose/wires
  for (int y = height - 40; y < height - 15; y++) {
    for (int x = width / 2 - 20; x < width / 2 + 20; x++) {
      uint16_t p = pixels[y * width + x];
      // RGB565 unpack: R(5), G(6), B(5)
      r_sum += (p >> 11) & 0x1F;
      g_sum += (p >> 5) & 0x3F;
      b_sum += p & 0x1F;
      safe_count++;
    }
  }

  if (safe_count == 0)
    return;
  int ref_r = r_sum / safe_count;
  int ref_g = g_sum / safe_count;
  int ref_b = b_sum / safe_count;

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
      uint16_t p = pixels[y * width + x];
      int r = (p >> 11) & 0x1F;
      int g = (p >> 5) & 0x3F;
      int b = p & 0x1F;

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

  // Output Telemetry: Require 40% of the active path to be a different color
  telemetryObstructed = ((float)obstacle_pixels / total_lookahead) > 0.40f;
  telemetryObstacleLeftRatio =
      total_left > 0 ? ((float)obstacle_left / total_left) : 0.0f;
  telemetryObstacleRightRatio =
      total_right > 0 ? ((float)obstacle_right / total_right) : 0.0f;

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
  camera_config_t config;
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
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 18000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QQVGA; // 160×120 — small & fast
  config.jpeg_quality = 10;            // 10 = high quality (lower = better)
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM; // requires Tools → PSRAM → Enabled

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAMERA] Init failed: 0x%x\n", err);
    return false;
  }

  // Improve image saturation & disable special effects
  sensor_t *s = esp_camera_sensor_get();
  s->set_special_effect(s, 0);
  s->set_saturation(s, 2);

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

  // Ensure telemetry is completely up-to-date for the current fetched frame
  if (visual_rgb_buf) {
    if (jpg2rgb565(fb->buf, fb->len, visual_rgb_buf, JPG_SCALE_NONE)) {
      analyzeFrame(visual_rgb_buf, fb->width, fb->height);
    }
  }

  server.sendHeader("X-Distance-CM", String(dist, 1));
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
                WiFi.localIP().toString() + "\",\"nav_state\":\"" +
                String(navStateName(navState)) + "\",\"command_id\":\"" +
                activeCommandId + "\",\"source\":\"" + activeCommandSource +
                "\",\"mode\":\"" + activeCommandMode + "\",\"safety_latched\":" +
                String(safetyStopLatched ? "true" : "false") +
                ",\"safety_reason\":\"" + safetyStopReason + "\"}";

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
  String source = extractJsonString(body, "source");
  String mode = extractJsonString(body, "mode");

  if (commandId.length() > 0)
    activeCommandId = commandId;
  if (source.length() > 0)
    activeCommandSource = source;
  if (mode.length() > 0)
    activeCommandMode = mode;

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
    String resp = "{\"ok\":true,\"command\":\"set_speed\",\"level\":\"" + level +
                  "\",\"base_speed\":" + String((int)currentSpeed) +
                  ",\"forward_speed\":" + String(forwardSpeed) +
                  ",\"safety\":" + boolStr + ",\"safety_reason\":\"" +
                  safetyReason + "\",\"command_id\":\"" + activeCommandId +
                  "\",\"source\":\"" + activeCommandSource +
                  "\",\"mode\":\"" + activeCommandMode + "\"}";
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(200, "application/json", resp);
    return;
  } else if (command == "set_light") {
    String level = extractJsonString(body, "level");
    uint8_t brightness = level.toInt();
    setHeadlight(brightness);

    String boolStr = safety_fired ? "true" : "false";
    String resp = "{\"ok\":true,\"command\":\"set_light\",\"level\":\"" + level +
                  "\",\"safety\":" + boolStr + ",\"safety_reason\":\"" +
                  safetyReason + "\",\"command_id\":\"" + activeCommandId +
                  "\",\"source\":\"" + activeCommandSource +
                  "\",\"mode\":\"" + activeCommandMode + "\"}";
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

    stopMotors();
    action = "stop";
    commandedAction = "stop";
    currentMoveAction = "stop";
    isMovingIndefinitely = false;
    navState = NAV_HOLD;
    String boolStr = safety_fired ? "true" : "false";
    String resp = "{\"ok\":true,\"action\":\"stop\",\"duration_ms\":0,\"safety\":" +
                  boolStr + ",\"safety_reason\":\"" + safetyReason +
                  "\",\"busy\":false,\"command_id\":\"" + activeCommandId +
                  "\",\"source\":\"" + activeCommandSource +
                  "\",\"mode\":\"" + activeCommandMode +
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

  float dist = readDistanceCM();

  if (command == "move_indefinitely") {
    String direction = extractJsonString(body, "direction");
    action = direction;
    duration_ms = 0;
    isMovingIndefinitely = (action != "stop");
    commandedAction = action;
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
  lastLeaseCommandTime = millis();
  navState = (action == "stop") ? NAV_HOLD : NAV_CRUISE;

  if (!isMovingIndefinitely && action != "stop") {
    targetStopTime = millis() + duration_ms;
  }

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

  String boolStr = safety_fired ? "true" : "false";
  String resp = "{\"ok\":true,\"action\":\"" + action +
                "\",\"duration_ms\":" + duration_ms + ",\"distance_cm\":\"" +
                String(dist, 1) + "\",\"safety\":" + boolStr +
                ",\"safety_reason\":\"" + safetyReason +
                "\",\"busy\":false,\"command_id\":\"" + activeCommandId +
                "\",\"source\":\"" + activeCommandSource +
                "\",\"mode\":\"" + activeCommandMode + "\",\"nav_state\":\"" +
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
  stopMotors();
  lastLeaseCommandTime = millis();
  Serial.println("[MOTORS] I2C ready");

  // Camera
  if (!initCamera()) {
    Serial.println("[ERROR] Camera init failed — halting");
    while (true)
      delay(1000);
  }

  // Allocate PSRAM buffer for RGB565 decoding
  visual_rgb_buf = (uint8_t *)ps_malloc(160 * 120 * 2);
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

  unsigned long now = millis();

  // Failsafe: any indefinite motion must be continuously renewed by middleware.
  if (isMovingIndefinitely && commandedAction != "stop" &&
      (now - lastLeaseCommandTime > COMMAND_LEASE_TIMEOUT_MS)) {
    stopMotors();
    currentMoveAction = "stop";
    commandedAction = "stop";
    isMovingIndefinitely = false;
    navState = NAV_HOLD;
    safetyStopLatched = true;
    safetyStopReason = "lease_timeout";
    Serial.printf(
        "[FAILSAFE] Motion lease expired after %lums without renew command. "
        "Motors stopped.\n",
        now - lastLeaseCommandTime);
  }

  // Reactive hard-stop only (no autonomous turn/backup planning in firmware).
  if (navState == NAV_CRUISE && currentMoveAction == "forward" &&
      (now - lastDistanceCheckTime >= DISTANCE_CHECK_INTERVAL_MS)) {
    lastDistanceCheckTime = now;

    bool blockedByVision = false;

    // Execute Edge-AI visual check if we have the buffer
    if (visual_rgb_buf) {
      camera_fb_t *fb = esp_camera_fb_get();
      if (fb) {
        // Decode JPEG to RGB565 directly into our PSRAM buffer
        if (jpg2rgb565(fb->buf, fb->len, visual_rgb_buf, JPG_SCALE_NONE)) {
          analyzeFrame(visual_rgb_buf, fb->width, fb->height);
          if (telemetryObstructed) {
            blockedByVision = true;
          }
        }
        esp_camera_fb_return(fb);
      }
    }

    if (blockedByVision) {
      stopMotors();
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

  if (currentMoveAction != "stop" && !isMovingIndefinitely &&
      now >= targetStopTime) {
    stopMotors();
    currentMoveAction = "stop";
    commandedAction = "stop";
    navState = NAV_HOLD;
  }
}
