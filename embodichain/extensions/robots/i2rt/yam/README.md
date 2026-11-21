# YAM dual-arm embodiment

`YamI2rtCfg` and `YamI2rt` own the converted dual-arm simulation model. The
registered ID `YamI2rt` resolves `robot_assets/YamI2rtV1.zip`, MD5
`23601ea820ff3e6260336c4962ef01a5`. Its URDF and collision meshes are used by
both simulation and kinematics.

The archive retains the pinned upstream source, conversion provenance and
I2RT Robotics MIT license. `scripts/tools/prepare_yam_i2rt_assets.py` owns the
source conversion; repository `NOTICE` and `licenses/YamI2rt-MIT.txt` preserve
attribution. Model gains and initial pose describe this simulation embodiment,
not verified hardware calibration. Task-specific contact settings and Newton
gravity compensation belong to the consuming embodiment or deployment.
