# Motion and planning, protocol 2

This version must be installed as a matched firmware/middleware pair. Middleware
claims a protocol 2 session before starting the camera/control tasks. Older
firmware is stopped by the compatibility probe and rejected; `/health` reports
the startup error. No automatic firmware upload is performed.

## Execution

| Component | Cadence | Responsibility |
|---|---|---|
| CameraStream | Capture, then 150 ms pause by default | Latest JPEG and timestamped obstacle telemetry |
| Local controller | 100 ms target; configurable 50–200 ms | Freshness, collision constraints, steering, executive, recovery |
| LLM | Existing provider rate limits, asynchronously | Scene, objective, up to three bounded skills |
| MotionSupervisor | One serialized network writer | Ownership, target changes, timed identity, 200 ms lease renewals |
| ESP32 motor task | 20 ms target | Fractional intensity easing, timed completion, independent watchdog |

Actual rates depend on camera, Wi-Fi and I2C latency. The controller never waits
for model completion or a recovery sleep. Camera failures leave the last frame
available only until its age limit. Reusing that frame does not create another
stall-detection sample. Obstacle-analysis age is checked separately from JPEG age.

Camera orientation is corrected with sensor vflip/hmirror for the existing
inverted mount. The dashboard does not rotate it again. Confirm left/right and
upright orientation on the actual hardware before tuning navigation.

## Continuous motion

All untimed middleware movement uses differential `drive` targets. Straight
forward and arcs use the same base intensity: slow 12, medium 20, fast 35. Reverse
and pivots use the corresponding unscaled levels 25, 40, 70. These are Nuvoton
actuator intensities, not measured velocities or PWM percentages.

The planner latches the maneuver's identity and steering bias, while updating
obstacle steering and the speed cap every tick. Steering changes are rate-limited.
Open-space reorientation uses forward arcs. Repeated identical wheel targets are
not resent; the existing target is renewed. Timed turns/reversals are submitted
once per maneuver ID, and observing completion cannot restart them.

Motor easing uses fractional state, a bounded intensity rate, and a bounded
change in that rate away from the terminal clamp. It is not a certified jerk
constraint on physical velocity: terminal clamping, integer output quantization,
motor deadband and wheel slip remain. Ordinary stops ease toward zero; emergency
stops zero the profile and motor outputs immediately. Camera/network work never
holds the motor mutex. Motor and headlight writes serialize access to I2C.

## Authority and stopping

Every `/move` request carries `session_id` and an increasing positive `sequence`.
The session is a 32-character UUID hex string. Start with:

```json
{"command":"claim","session_id":"0123456789abcdef0123456789abcdef","sequence":1}
```

The response must contain `{"ok":true,"protocol":2}`. Claiming stops existing
motion. It is an ordering/session mechanism, not authentication; this is a trusted
local-network API. A different session or older sequence is rejected before
actuator mutations. Existing movement forms remain available inside this envelope.

```json
{"command":"drive","left":20,"right":20,"command_id":"move-1","source":"planner","mode":"explore","session_id":"0123456789abcdef0123456789abcdef","sequence":2}
```

Renewals carry the active motion's identity and never carry wheel targets:

```json
{"command":"renew","target_id":"move-1","command_id":"renew-1","source":"lease","mode":"explore","session_id":"0123456789abcdef0123456789abcdef","sequence":3}
```

An expired, stopped or superseded motion cannot be revived by renewal. Both
continuous and timed movements have a 750 ms firmware watchdog. A timed motion
also has its own deadline (maximum 3 seconds); renewal never extends that deadline.
Forward motion stops if firmware image analysis has not succeeded within 800 ms,
even if the HTTP server is busy. Visual obstacle detection defaults on; its
color-based classifier is not a depth sensor and remains configurable.

Middleware stops renewing autonomous motion after 800 ms without fresh control
work. Remote inputs expire after 700 ms; ownership remains manual until the
configured 3-second inactivity limit. Remote takeover cancels recovery within the
same actuator lock. Queue leave/session expiry explicitly releases authority.
The lock may wait on one bounded 350 ms HTTP operation; firmware watchdog checks
run independently. These are software timing targets, not measured stopping distances.

## LLM skill contract

The existing scene response can include:

```json
{
  "strategy": {
    "objective": "Explore the clear floor to the left, then observe",
    "steps": [
      {"skill":"explore","direction":"left","duration_s":4},
      {"skill":"pause","direction":"forward","duration_s":1}
    ]
  }
}
```

Supported skills: `explore` (forward/left/right), `follow_boundary` (left/right),
`scan` (left/right), `pause` (direction omitted or forward). Durations are 0.5–6
seconds, with scans capped at 1 second. Maximum three steps. Unknown skills,
invalid directions and non-finite/out-of-range durations reject the entire strategy.
No model-supplied motor intensity, metric coordinate or untracked landmark is executed.

The executive validates current local conditions, adopts revisions at maneuver
boundaries, and cancels on hard/blocked evidence, boundary loss, expiry, recovery,
or human takeover. Results are tied to source capture time and a planning
generation, which changes on persona/authority/recovery transitions. Strategies
expire at the original capture time plus the semantic TTL. LLM timeout does not
command a pause or scan; valid skills continue within their TTL and local planning
fills gaps. Semantic fallback updates from current telemetry each tick.

Execution history is included in subsequent LLM prompts and dashboard telemetry.
`budget_complete` means a skill exhausted its time budget, not that a spatial goal
was reached. This is egocentric exploration with short execution memory. Accurate
landmark approach, room mapping and return-to-location still need tracking,
calibration and localization/odometry.

## Validation and floor tuning

```bash
cd middleware
.venv/bin/python -m unittest discover -q
```

From the repository root:

```bash
clang++ -std=c++17 -Wall -Wextra -Werror firmware/tests/motion_profile_test.cpp -o /tmp/wheelson-profile-test
/tmp/wheelson-profile-test
arduino-cli compile --fqbn esp32:esp32:esp32:PartitionScheme=huge_app,PSRAM=enabled firmware/wheelson_explorer
```

The Python suite includes command/renewal races, timed deduplication, remote
takeover, stale observations, skill validation, bounded recovery, and a complete
mocked exploration loop. The native C++ test checks starts, reversals, stops,
fractional increments and delayed motor ticks. Compilation does not flash hardware.

After flashing the matched firmware, check the camera orientation and left/right
motor directions with wheels raised. Then test on a clear floor at slow speed:
straight-to-arc continuity; a visible obstacle; a model timeout; camera/network
loss; takeover during each recovery phase. Record SSE `wheel_target`, `wheel_output`,
`perception_age_s`, command IDs and `executive` outcomes. Wheel output telemetry
reports commanded intensity, not measured wheel speed. Adjust deadband, easing and
obstacle thresholds from those trials before increasing cruising speed.
