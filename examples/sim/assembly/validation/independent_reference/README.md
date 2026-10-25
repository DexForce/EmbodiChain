# Corrected independent contact-profile comparison

The measurements in `invalid_global_contact_defaults/` and the earlier parent circle sweep are superseded. The old analytic peg inherited MuJoCo global contact solref20ms while plate geoms used the nested mjfit2ms default. Corrected XML explicitly assigns the source plate solref, solimp, solmix and priority to the replacement primitive peg. Frictionless condim1 is intentional for both parts.

Only the new `mujoco_3.12.0.json` and `mujoco_3.14.0.json` represent matched contact profiles. Accuracy gates use worst settled error and settled-window drift; thresholds are unchanged.
