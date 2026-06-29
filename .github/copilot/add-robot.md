# Add Robot for GitHub Copilot

Canonical source: `.agents/skills/add-robot/`

Use this adapter when adding a new robot to EmbodiChain or adding a variant to an
existing robot (a new version / arm kind / hand brand). Then follow
`.agents/skills/add-robot/SKILL.md` for the `RobotCfg` protocol (`_build_defaults`
hook + `build_pk_serial_chain` + inherited serialization), the single-file vs
package layouts, and the scaffold steps.
