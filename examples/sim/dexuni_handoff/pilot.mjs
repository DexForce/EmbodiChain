// Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { randomUUID } from "node:crypto";

const [directory, pythonPrefix, library, sdk] = process.argv.slice(2);
const { Run, loadSettings } = await import(
  pathToFileURL(path.resolve(library)).href
);
const [record] = fs
  .readdirSync(directory)
  .filter((n) => /^dsws_.*\.json$/.test(n));
const session = record.slice(0, -5);
const runId = "dexuni-" + randomUUID();
const settings = loadSettings();
settings.session_directory = directory;
settings.agent_python = pythonPrefix;
settings.agent_sdk = sdk;
settings.world_restarts = 0;
function configure(turn) {
  settings.mcp_servers = [
    {
      name: "embodichain",
      command: path.join(pythonPrefix, "bin/python"),
      args: [
        "-m",
        "embodichain",
        "mcp",
        "--session",
        "{session_id}",
        "--session-directory",
        "{session_directory}",
        "--handoff-key-file",
        path.resolve(directory, "writer.key"),
        "--run-id",
        runId,
        "--turn",
        String(turn),
      ],
      env: [],
    },
  ];
}
const binding = {
  arena: "default",
  robots: {
    UR5: { robot_cfg: "URRobot", variant: "ur5", object_ref: "arm_b" },
  },
  bodies: {},
  physics_dt: 0.01,
  steps_per_control: 4,
};
const task =
  "Use the mounted DexSim and EmbodiChain MCP tools only. You are attached to a manual DexUni World with an existing floor. List objects. Import UR5.urdf twice through dexsim_import_urdf: name arm_a at [0,0,0], arm_b at [2,0,0], both fixed_base=true. Rebuild with dexsim_rebuild_scene. Inspect solver and native joint state of both robots. Use embodichain_inspect_handoff; then call embodichain_bind_world with the returned generation and topology_revision and this exact binding: " +
  JSON.stringify(binding) +
  ". Call embodichain_control_status and use its session_id, revision and topology_revision in embodichain_control_target_and_step. Supply command_id='one', robot_uid='UR5', control_part='arm', target=[0.08,-0.06,0.04,-0.03,0.02,-0.01], units=['rad','rad','rad','rad','rad','rad'], control_interval_s=0.04. Read both robots with the native tools. Do not release until the next turn.";
const opts = {
  onPermission: (r) =>
    /dexsim_|embodichain_/.test(r.toolCall?.title ?? "")
      ? (r.options.find((o) => o.kind === "allow_once")?.optionId ?? null)
      : null,
};
configure(1);
const run = await Run.create(settings, {
  task,
  session_id: session,
  run_id: runId,
  agent: "dsh",
  export_api: false,
  wall_seconds: 360,
  max_model_requests: 30,
});
console.log(runId);
try {
  await run.send(task, opts);
} finally {
  await run.close();
}
function entries() {
  return fs
    .readFileSync(path.join(directory, session + ".jsonl"), "utf8")
    .trim()
    .split("\n")
    .map(JSON.parse);
}
const first = entries().find(
  (e) =>
    e.status === "ok" &&
    e.client?.run_id === runId &&
    e.params?.arguments?.operation === "target_and_step",
);
if (!first) throw Error("First turn did not issue the target");
configure(2);
const second = Run.open(settings, runId);
try {
  await second.send(
    "Use only the mounted MCP tools. Resubmit the same EmbodiChain target_and_step with the identical arguments " +
      JSON.stringify(first.params.arguments.arguments) +
      ". Query control_receipt for one, then control_stop. Inspect the current handoff generation, release via embodichain_release_world, and use the native DexSim tools to read both robots. Report actual clock and measured B joint1.",
    opts,
  );
} finally {
  await second.close();
}
const events = entries();
const turn2 = events.filter(
  (e) =>
    e.client?.run_id === runId && e.client?.turn === 2 && e.status === "ok",
);
if (
  !turn2.some((e) => e.params?.arguments?.operation === "target_and_step") ||
  !turn2.some((e) => e.params?.arguments?.arguments?.operation === "release")
)
  throw Error("Reconnect/release evidence missing");
console.log("Completed", runId, session);
