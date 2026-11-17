# Configuration-driven Peg policy and inference recording

The frozen update-300 model was trained from random weights with seed 0.
Source evaluation: **7930/8192 (96.8018%)**; isolated installed wheel:
**7933/8192 (96.8384%)**. Both exceed the 95% engineering target.
The test protocol uses deterministic means, seed 20000 and 1024 arenas with
8 scored completions per arena. Validation selected the model before testing.

- [Complete checkpoint, portable configs/resources and raw evidence](configured-peg.zip)
- [Native learned-policy inference video, 1280×720 /25 fps](preview.mp4)
- [GIF preview](preview.gif)
- [Reset/terminal contact sheet](contact_sheet.png)
- [Training and online validation curves](learning-curves.png)
- [Machine-readable protocol and all recorded Peg executions](reproduction.json)

The current-code video retains seeds 30000, 30001 and 30002 in order: 3/3
successes. It contains actual native RGB and the deterministic trained policy,
with grasp, penetration and stable-seat checks enabled. Terminal pauses hold
physics fixed. This three-episode recording is separate from success-rate
qualification.

The development profile uses one geometry and one training seed with public,
uncalibrated robot dynamics. It is not a real-device or contact-precision
qualification. SDK binaries and dependency patches are not included.
