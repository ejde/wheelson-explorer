// Finch face renderer for the Wheelson ST7735S display.
// Draws into a PSRAM sprite, then pushes the whole frame at once so the
// face never tears mid-draw. Runs at ~10 fps in its own FreeRTOS task.
#define LGFX_USE_V1
#include <LovyanGFX.hpp>
#include <Wire.h>
#include <freertos/semphr.h>
#include "face.h"

// I2C bus mutex owned by the main sketch (also guards motor commands).
extern SemaphoreHandle_t i2cMutex;

// ---- LCD device: panel config copied from CircuitMess WheelsonDisplay ----
class FaceLCD : public lgfx::LGFX_Device {
 public:
  FaceLCD() {
    {
      auto bus = new lgfx::Bus_SPI();
      auto cfg = bus->config();
      cfg.spi_host = VSPI_HOST;
      cfg.spi_mode = 0;
      cfg.freq_write = 27000000;
      cfg.freq_read = 27000000;
      cfg.spi_3wire = false;
      cfg.use_lock = true;
      cfg.dma_channel = 1;
      cfg.pin_sclk = 2;
      cfg.pin_mosi = 13;
      cfg.pin_miso = -1;
      cfg.pin_dc = 12;
      bus->config(cfg);
      _bus = bus;
    }
    {
      auto panel = new lgfx::Panel_ST7735S();
      auto cfg = panel->config();
      cfg.pin_cs = 32;
      cfg.pin_rst = -1;
      cfg.pin_busy = -1;
      cfg.memory_width = 128;
      cfg.memory_height = 160;
      cfg.panel_width = 128;
      cfg.panel_height = 160;
#ifdef FACE_PANEL2
      cfg.offset_x = -2;
      cfg.offset_y = 1;
      cfg.rgb_order = true;
#else
      cfg.offset_x = 0;
      cfg.offset_y = 0;
      cfg.rgb_order = false;
#endif
      cfg.offset_rotation = 6;
      cfg.readable = false;
      cfg.invert = false;
      cfg.dlen_16bit = false;
      cfg.bus_shared = false;
      panel->setBus(_bus);
      panel->config(cfg);
      setPanel(panel);
    }
  }
  lgfx::Bus_SPI* _bus = nullptr;
};

static FaceLCD lcd;
static lgfx::LGFX_Sprite sprite(&lcd);

// ---- shared face state (guarded by a mutex) ----
static SemaphoreHandle_t faceMutex = nullptr;
static FaceExpr autoExpr = FACE_IDLE;
static String autoStatus = "HOLD";
static FaceExpr ovExpr = FACE_IDLE;
static String ovStatus = "";
static int8_t ovLookX = 0, ovLookY = 0;
static uint32_t ovUntil = 0;  // millis() deadline, 0 = no override
static bool ovActive = false;

void faceSetAutoExpr(FaceExpr e) {
  if (!faceMutex) return;
  xSemaphoreTake(faceMutex, portMAX_DELAY);
  autoExpr = e;
  xSemaphoreGive(faceMutex);
}

void faceSetAutoStatus(const String& s) {
  if (!faceMutex) return;
  xSemaphoreTake(faceMutex, portMAX_DELAY);
  autoStatus = s;
  autoStatus.remove(24);
  xSemaphoreGive(faceMutex);
}

void faceSetOverride(FaceExpr e, const String& s, int8_t dx, int8_t dy,
                     uint32_t holdMs) {
  if (!faceMutex) return;
  xSemaphoreTake(faceMutex, portMAX_DELAY);
  ovExpr = e;
  ovStatus = s;
  ovStatus.remove(24);
  ovLookX = constrain(dx, -10, 10);
  ovLookY = constrain(dy, -10, 10);
  ovUntil = millis() + holdMs;
  ovActive = true;
  xSemaphoreGive(faceMutex);
}

// ---- drawing ----
static uint16_t C(uint8_t r, uint8_t g, uint8_t b) {
  return sprite.color565(r, g, b);
}

// Draw one frame of the Finch face into the sprite.
static void drawFace(FaceExpr expr, const String& status, int lookX,
                     int lookY, bool blink) {
  const int W = sprite.width();
  const int H = sprite.height();
  const uint16_t bg = C(10, 16, 32);
  const uint16_t bar = C(18, 28, 52);
  const uint16_t white = C(245, 245, 250);
  const uint16_t dark = C(12, 12, 16);
  const uint16_t orange = C(255, 160, 0);
  const uint16_t slate = C(70, 90, 130);

  sprite.fillScreen(bg);

  // Ear tufts.
  sprite.fillTriangle(28, 20, 44, 20, 36, 4, slate);
  sprite.fillTriangle(W - 44, 20, W - 28, 20, W - 36, 4, slate);

  const int cy = 54;
  const int cx1 = W / 2 - 28;
  const int cx2 = W / 2 + 28;
  const int exRy = (expr == FACE_EXPLORE) ? 30 : (expr == FACE_STUCK) ? 22 : 27;

  if (expr == FACE_HAPPY) {
    // Happy arc eyes: two "^^" polylines, drawn thick.
    for (int t = -1; t <= 1; t++) {
      for (int cx : {cx1, cx2}) {
        sprite.drawLine(cx - 15, cy + 8 + t, cx, cy - 10 + t, white);
        sprite.drawLine(cx, cy - 10 + t, cx + 15, cy + 8 + t, white);
      }
    }
  } else if (blink) {
    // Closed lids: a dark line across each eye.
    for (int cx : {cx1, cx2}) {
      sprite.fillEllipse(cx, cy, 24, exRy, bg);
      sprite.fillRect(cx - 20, cy - 1, 40, 3, dark);
    }
  } else {
    // Eye whites, pupils (+glint), then chunky glasses frames.
    for (int cx : {cx1, cx2}) {
      sprite.fillEllipse(cx, cy, 24, exRy, white);
      int px = cx + lookX;
      int py = cy + lookY;
      sprite.fillCircle(px, py, 10, dark);
      sprite.fillCircle(px - 3, py - 3, 3, white);
    }
  }

  // Chunky black glasses: thick rounded frames + bridge + temple arms.
  for (int k = 0; k < 5; k++) {
    sprite.drawRoundRect(cx1 - 30 + k, 20 + k, 60 - 2 * k, 68 - 2 * k, 12, dark);
    sprite.drawRoundRect(cx2 - 30 + k, 20 + k, 60 - 2 * k, 68 - 2 * k, 12, dark);
  }
  sprite.fillRect(0, 50, 22, 6, dark);
  sprite.fillRect(W - 22, 50, 22, 6, dark);

  // Worried brows when stuck.
  if (expr == FACE_STUCK) {
    sprite.drawLine(cx1 - 24, 12, cx1 + 4, 20, white);
    sprite.drawLine(cx1 - 24, 13, cx1 + 4, 21, white);
    sprite.drawLine(cx2 - 4, 20, cx2 + 24, 12, white);
    sprite.drawLine(cx2 - 4, 21, cx2 + 24, 13, white);
  }

  // Beak.
  sprite.fillTriangle(W / 2 - 8, 88, W / 2 + 8, 88, W / 2, 100, orange);

  // Status bar.
  sprite.fillRect(0, H - 20, W, 20, bar);
  sprite.setTextFont(2);
  sprite.setTextColor(white, bar);
  int tw = sprite.textWidth(status.c_str());
  sprite.drawString(status.c_str(), (W - tw) / 2, H - 15);
}

static void faceTaskFn(void*) {
  uint32_t nextBlink = millis() + 2500;
  uint32_t blinkEnd = 0;
  uint32_t nextWander = 0;
  int wanderX = 0, wanderY = 0;
  int curX = 0, curY = 0;
  for (;;) {
    FaceExpr expr;
    String status;
    int lookX, lookY;
    uint32_t now = millis();
    xSemaphoreTake(faceMutex, portMAX_DELAY);
    if (ovActive && (int32_t)(ovUntil - now) > 0) {
      expr = ovExpr;
      status = ovStatus.length() ? ovStatus : autoStatus;
      lookX = ovLookX;
      lookY = ovLookY;
    } else {
      ovActive = false;
      expr = autoExpr;
      status = autoStatus;
      if (expr == FACE_THINK) {
        lookX = 4;
        lookY = -6;
      } else {
        if (now >= nextWander) {
          wanderX = (int)(esp_random() % 9) - 4;
          wanderY = (int)(esp_random() % 7) - 3;
          nextWander = now + 2200 + (esp_random() % 1800);
        }
        if (curX < wanderX) curX++;
        if (curX > wanderX) curX--;
        if (curY < wanderY) curY++;
        if (curY > wanderY) curY--;
        lookX = curX;
        lookY = curY;
      }
    }
    xSemaphoreGive(faceMutex);

    bool blink = false;
    if (now >= blinkEnd && now >= nextBlink) {
      blink = true;
      blinkEnd = now + 130;
      nextBlink = now + 2300 + (esp_random() % 2200);
    } else if (now < blinkEnd) {
      blink = true;
    }

    drawFace(expr, status, constrain(lookX, -8, 8), constrain(lookY, -8, 8),
             blink);
    sprite.pushSprite(0, 0);
    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

void faceInit() {
  faceMutex = xSemaphoreCreateMutex();
  lcd.init();
  lcd.setRotation(1);  // landscape: 160x128
  sprite.setPsram(true);
  if (!sprite.createSprite(lcd.width(), lcd.height())) {
    Serial.println("[FACE] sprite alloc failed");
  }
  // Display backlight on via the Nuvoton co-processor (0x38, {0x20, 1}).
  if (i2cMutex) {
    xSemaphoreTake(i2cMutex, portMAX_DELAY);
    Wire.beginTransmission(0x38);
    Wire.write(0x20);
    Wire.write(0x01);
    Wire.endTransmission();
    xSemaphoreGive(i2cMutex);
  }
  sprite.fillScreen(0);
  sprite.pushSprite(0, 0);
  if (xTaskCreate(faceTaskFn, "face", 8192, nullptr, 1, nullptr) != pdPASS) {
    Serial.println("[FACE] task create failed");
  } else {
    Serial.println("[FACE] Finch face task started");
  }
}
