#include "../wheelson_explorer/motion_profile.h"
#include <cassert>
#include <iostream>

int main() {
  MotionProfile p;
  float previous = 0;
  for (int i = 0; i < 300; ++i) {
    p.step(40, .02f);
    assert(p.value >= previous && p.value <= 40);
    assert(p.value - previous <= 2.001f);
    previous = p.value;
  }
  assert(std::abs(p.value - 40) < .01f);
  // Reversals cross zero without jumping directly to reverse output.
  for (int i = 0; i < 300; ++i) {
    float before = p.value;
    p.step(-40, .02f);
    assert(p.value >= -40 && p.value <= 40);
    assert(std::abs(p.value - before) <= 2.001f);
  }
  assert(std::abs(p.value + 40) < .01f);
  p.reset();
  assert(p.value == 0 && p.rate == 0);
  p.step(20, .001f);
  assert(p.value > 0); // Fractional increments survive integer PWM quantization.
  float before = p.value;
  p.step(20, 0);
  assert(p.value == before);
  p.step(127, 10); // A delayed scheduler tick cannot skip the easing profile.
  assert(p.value - before <= 2.001f);
  for (int i = 0; i < 300; ++i) p.step(0, .02f);
  assert(std::abs(p.value) < .01f);
  std::cout << "Motion profile: starts, reversals, stops, fractional ticks and delayed ticks passed\n";
}
