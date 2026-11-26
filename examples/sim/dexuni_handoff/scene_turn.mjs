// Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [root, python, entry, phase, runId] = process.argv.slice(2);
const { Run, loadSettings } = await import(
  pathToFileURL(path.resolve(entry)).href
);
const folder = path.join(root, "sessions"),
  m = JSON.parse(fs.readFileSync(path.join(root, "scene.json")));
const settings = loadSettings(),
  turn = Number(phase),
  key = path.join(root, "writer.key");
settings.session_directory = folder;
settings.agent_sdk = path.join(python, "lib/python3.11/site-packages");
settings.agent_python = python;
settings.world_restarts = 0;
settings.mcp_servers = [
  {
    name: "embodichain",
    command: path.join(python, "bin/python"),
    args: [
      "-m",
      "embodichain",
      "mcp",
      "--session",
      "{session_id}",
      "--session-directory",
      "{session_directory}",
      "--handoff-key-file",
      key,
      "--motion-output",
      path.join(root, "jobs"),
      "--run-id",
      runId,
      "--turn",
      phase,
    ],
    env: [],
  },
];
const allowed = {
  onPermission: (r) =>
    /embodichain_|dexsim_/.test(r.toolCall?.title ?? "")
      ? (r.options.find((o) => o.kind === "allow_once")?.optionId ?? null)
      : null,
};
const locomotion = m.binding.profile !== "ur5-pick-place";
let text;
if (turn === 1)
  text = m.authoring.length
    ? "Use only DexSim MCP. This manual DexUni World contains only a floor. Build the scene yourself by issuing all these ordered calls, each [native method, arguments]. Map each to the corresponding DexSim MCP tool (scene.create_primitive=dexsim_create_primitive, articulation.set_state=dexsim_set_articulation_state, articulation.set_joint_state=dexsim_set_joint_state, simulation.step=dexsim_step_simulation). Do not change parameter values. Use dexsim_import_usd for scene.import_usd, dexsim_create_camera for scene.create_camera and dexsim_create_light for light.create. Do not use scripts, import a whole scene, create a World, or advance any extra steps. Then inspect the robot and task objects to verify your scene. Stop at that boundary.\n" +
      JSON.stringify(m.authoring)
    : "Use only DexSim MCP to inspect the already prepared World: list its objects, read the selected robot and task objects. Do not mutate or step.";
if (turn === 2 && locomotion)
  text =
    "Use only the mounted MCP tools. Inspect the handoff and use its generation/topology to bind exactly " +
    JSON.stringify(m.binding) +
    ". Use embodichain_start_policy(command_id=policy-1) and get_run with wait_seconds=20 until done. Report physical verdict, metrics and artifact references. Finally release the World and verify the authoring phase. Do not rebuild/reset or step outside the job.";
if (turn === 2 && !locomotion)
  text =
    "Use only EmbodiChain MCP. Call inspect_handoff and use the returned generation and topology_revision to bind this exact role mapping: " +
    JSON.stringify(m.binding) +
    ". Call list_capabilities; submit start_skill skill_id=pick_up object_ref=red, command_id=pick. Poll get_run using wait_seconds=20 until done. Return the actual grasp_ref. Do not place yet.";
if (turn === 3) {
  const receipts = fs
    .readdirSync(path.join(root, "jobs"))
    .filter((n) => n.startsWith("ecmotion_"))
    .map((n) =>
      JSON.parse(fs.readFileSync(path.join(root, "jobs", n, "receipt.json"))),
    );
  const pick = receipts.find(
    (r) => r.command_id === "pick" && r.status === "succeeded",
  );
  if (!pick) throw Error("Pick did not pass");
  text =
    "Use only the configured MCP tools. Revalidate grasp " +
    pick.result.grasp_ref +
    " on the original World. The user's destination chosen now is right. Submit start_skill(skill_id=place,object_ref=red,target_ref=right,grasp_ref=" +
    pick.result.grasp_ref +
    ",command_id=place). Use get_run with wait_seconds=20 until done. Report all physical checks. Release the binding using inspect_handoff and release_world once the job is terminal.";
}
const run =
  turn === 1
    ? await Run.create(settings, {
        run_id: runId,
        session_id: m.session_id,
        task: text,
        agent: "dsh",
        export_api: false,
        max_model_requests: 60,
        wall_seconds: 900,
      })
    : Run.open(settings, runId);
try {
  const r = await run.send(text, allowed);
  if (r.end_reason !== "replied") throw Error(r.end_reason);
  console.log(r.reply);
} finally {
  await run.close();
}
