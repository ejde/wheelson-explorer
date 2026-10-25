// Finch face for the Wheelson display.
//
// Hardware (from CircuitMess Wheelson-Library, WheelsonDisplay.cpp):
//   ST7735S panel, 128x160, SPI VSPI_HOST, SCLK GPIO 2, MOSI GPIO 13,
//   DC GPIO 12, CS GPIO 32, no MISO, no reset pin, 27 MHz, SPI mode 0.
//   FACE_PANEL2 selects the v2/v3 panel variant (rgb_order + offsets).
//   NOTE: LCD MOSI/DC share GPIO 13/12 with the (disabled-by-default)
//   ultrasonic TRIG/ECHO pins -- do not enable the ultrasonic sensor
//   while the face display is in use.
// Backlight is switched on via the Nuvoton co-processor over I2C
// (address 0x38, message {0x20, 0x01}), same bus the motor task uses.
#pragma once

#include <Arduino.h>

enum FaceExpr : uint8_t {
  FACE_IDLE,    // neutral, slow blink
  FACE_EXPLORE, // alert, eyes slightly wider
  FACE_THINK,   // pupils glance up
  FACE_STUCK,   // narrowed eyes + worried brows
  FACE_HAPPY    // happy arc eyes
};

// Start the display + face task. Call once from setup(), after Wire.begin()
// and after the I2C mutex exists.
void faceInit();

// Automatic state, driven by the sketch loop() from nav/safety state.
// The face task prefers an active override (see below) over these.
void faceSetAutoExpr(FaceExpr e);
void faceSetAutoStatus(const String& s);

// One-shot override, e.g. from POST /face (middleware persona pushes).
// Holds for holdMs, then the face falls back to the automatic state.
void faceSetOverride(FaceExpr e, const String& s, int8_t lookX, int8_t lookY,
                     uint32_t holdMs);
