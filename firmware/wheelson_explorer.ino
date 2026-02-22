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
 * Distance sensor: HC-SR04 wired to TRIG_PIN / ECHO_PIN (see config).
 * Hard-stop: any "forward" command is overridden to "stop" when the
 * sensor reports < HARD_STOP_CM centimetres.
 *
 * ─── Arduino IDE setup ───────────────────────────────────────────────
 *  Board      : ESP32 Dev Module
 *  Partition  : Huge APP (3MB No OTA/1MB SPIFFS)
 *  No external libraries required — uses ESP32 Arduino core only.
 */

#include "esp_camera.h"
#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>
#include <Wire.h>

// ─── USER CONFIGURATION ──────────────────────────────────────────────────────

#define WIFI_SSID "WIFI_SSID"
#define WIFI_PASSWORD "WIFI_PASSWORD"

// HC-SR04 distance sensor — change if your wiring differs
#define TRIG_PIN 13
#define ECHO_PIN 12

// Robot stops if obstacle is closer than this (cm)
#define HARD_STOP_CM 10

// Motor drive intensity: 0–127 (signed int8_t range used by Nuvoton)
#define MOTOR_SPEED 90

// ─── NUVOTON I2C MOTOR CONTROLLER ────────────────────────────────────────────
// Protocol reverse-engineered from
// Wheelson-Library/src/Nuvoton/WheelsonMotor.cpp
#define NUVOTON_ADDR 0x38   // Nuvoton co-processor I2C address
#define MOTOR_SET_BYTE 0x30 // Command byte to set motor intensity
#define I2C_SDA 14          // Wheelson I2C bus (from Pins.hpp)
#define I2C_SCL 15

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

void stopMotors() {
  for (uint8_t i = 0; i < 4; i++)
    setMotor(i, 0);
}

// Drive all four wheels forward
void driveForward() {
  setMotor(MOTOR_FL, MOTOR_SPEED);
  setMotor(MOTOR_BL, MOTOR_SPEED);
  setMotor(MOTOR_FR, MOTOR_SPEED);
  setMotor(MOTOR_BR, MOTOR_SPEED);
}

// Drive all four wheels backward
void driveBackward() {
  setMotor(MOTOR_FL, -MOTOR_SPEED);
  setMotor(MOTOR_BL, -MOTOR_SPEED);
  setMotor(MOTOR_FR, -MOTOR_SPEED);
  setMotor(MOTOR_BR, -MOTOR_SPEED);
}

// Left wheels backward, right wheels forward → pivot left
void turnLeft() {
  setMotor(MOTOR_FL, -MOTOR_SPEED);
  setMotor(MOTOR_BL, -MOTOR_SPEED);
  setMotor(MOTOR_FR, MOTOR_SPEED);
  setMotor(MOTOR_BR, MOTOR_SPEED);
}

// Left wheels forward, right wheels backward → pivot right
void turnRight() {
  setMotor(MOTOR_FL, MOTOR_SPEED);
  setMotor(MOTOR_BL, MOTOR_SPEED);
  setMotor(MOTOR_FR, -MOTOR_SPEED);
  setMotor(MOTOR_BR, -MOTOR_SPEED);
}

// ─── DISTANCE SENSOR ─────────────────────────────────────────────────────────

/**
 * Trigger HC-SR04 pulse and measure echo time.
 * Returns distance in centimetres, or 999.0 on timeout (clear path).
 */
float readDistanceCM() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // 30 ms timeout → max ~515 cm, practical open-room limit
  long duration = pulseIn(ECHO_PIN, HIGH, 30000UL);
  if (duration == 0)
    return 999.0f;

  return duration * 0.0343f / 2.0f;
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

  server.sendHeader("X-Distance-CM", String(dist, 1));
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
                WiFi.localIP().toString() + "\"}";

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
  String action = extractJsonString(body, "action");
  int duration_ms = extractJsonInt(body, "duration_ms");

  if (action.length() == 0)
    action = "stop";
  if (duration_ms <= 0)
    duration_ms = 500;
  float dist = readDistanceCM();
  bool safety_fired = false;

  // ⚠️  HARD STOP: override forward if obstacle is too close
  if (dist < HARD_STOP_CM && action == "forward") {
    Serial.printf("[SAFETY] Hard stop! dist=%.1f cm < %d cm\n", dist,
                  HARD_STOP_CM);
    action = "stop";
    safety_fired = true;
  }

  // Drive
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

  // Let the robot move, then brake
  if (action != "stop") {
    delay(duration_ms);
    stopMotors();
  }

  String boolStr = safety_fired ? "true" : "false";
  String resp = "{\"ok\":true,\"action\":\"" + action +
                "\",\"duration_ms\":" + duration_ms + ",\"distance_cm\":\"" +
                String(dist, 1) + "\",\"safety\":" + boolStr + "}";

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

  // Distance sensor
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);

  // Nuvoton co-processor reset (MUST happen before Wire.begin + motor cmds)
  nuvotonBegin();

  // I2C → Nuvoton motor controller
  Wire.begin(I2C_SDA, I2C_SCL);
  stopMotors();
  Serial.println("[MOTORS] I2C ready");

  // Camera
  if (!initCamera()) {
    Serial.println("[ERROR] Camera init failed — halting");
    while (true)
      delay(1000);
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

void loop() { server.handleClient(); }
