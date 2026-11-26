// Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
import fs from "node:fs";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [directoryArg, pythonArg, output, entry] = process.argv.slice(2);
if (!output)
  throw new Error(
    "Usage: node dexpilot.mjs HOST_DIR PYTHON REPORT [DEXPILOT_ENTRY]",
  );
const directory = path.resolve(directoryArg);
const python = path.resolve(pythonArg);
const { Run, loadSettings } = await import(
  entry ? pathToFileURL(path.resolve(entry)).href : "dexpilot"
);
const records = fs
  .readdirSync(directory)
  .filter((name) => /^dsws_.*\.json$/.test(name));
if (records.length !== 1)
  throw new Error("Expected exactly one dedicated Session");
const sessionId = records[0].slice(0, -5);
const settings = loadSettings();
settings.session_directory = directory;
settings.agent_sdk = null;
settings.world_restarts = 0;

// This DexPilot revision supplies a per-Run ACP server factory for DSH.
// Extend this instance's tool list only, without changing the installed CLI.
function register(run, turn) {
  if (typeof run.mcpServers !== "function")
    throw new Error("Unsupported DexPilot MCP factory");
  run.mcpServers = () => [
    {
      name: "embodichain",
      command: python,
      args: [
        "-m",
        "embodichain",
        "mcp",
        "--session",
        sessionId,
        "--session-directory",
        directory,
        "--writer-key-file",
        path.join(directory, "writer.key"),
        "--run-id",
        run.runId,
        "--turn",
        String(turn),
      ],
      env: [],
    },
  ];
}
function nativeReadback(label) {
  const file = path.resolve(output + "." + label + ".json");
  execFileSync(
    python,
    ["-m", "examples.sim.attached_world.readback", directory, file],
    { stdio: "inherit" },
  );
  return JSON.parse(fs.readFileSync(file, "utf8"));
}
const resumeId = process.env.DEXPILOT_RESUME_RUN;
const before = resumeId
  ? JSON.parse(fs.readFileSync(output + ".before.json", "utf8"))
  : nativeReadback("before");
if (before.control.state !== "ready" || before.control.control_step !== 0)
  throw new Error("Requires fresh single-writer fixture");
const args = {
  command_id: "target-b-once",
  session_id: sessionId,
  topology_revision: before.control.topology_revision,
  revision: before.control.revision,
  robot_uid: "UR5",
  control_part: "arm",
  target: [0.15, -0.15, 0.12, -0.15, 0.12, -0.15],
  units: Array(6).fill("rad"),
  control_interval_s: 0.04,
  tolerance: 0.001,
};
const task =
  "Call the registered MCP tool embodichain_control_target_and_step exactly once with these exact arguments: " +
  JSON.stringify(args) +
  ". The explicit operator-selected UID UR5 is bound to robot_b. Return the tool's receipt. Do not call other tools or stop the writer.";
const run = resumeId
  ? Run.open(settings, resumeId)
  : await Run.create(settings, {
      task,
      session_id: sessionId,
      agent: "dsh",
      export_api: false,
      wall_seconds: 180,
      max_model_requests: 12,
    });
const opts = {
  onPermission: (request) => {
    // Allow only the operator-selected MCP tools, never shell/file operations.
    const name = JSON.stringify(request.toolCall?.title ?? "");
    return name.includes("embodichain_control_")
      ? (request.options.find((o) => o.kind === "allow_once")?.optionId ?? null)
      : null;
  },
};
register(run, 1);
if (!resumeId) {
  try {
    await run.send(task, opts);
  } finally {
    await run.close();
  }
}
const journalPath = path.join(directory, sessionId + ".jsonl");
function entries() {
  return fs
    .readFileSync(journalPath, "utf8")
    .trim()
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}
const first = entries().find(
  (e) =>
    e.client?.run_id === run.runId &&
    e.client?.turn === 1 &&
    e.params?.arguments?.operation === "target_and_step" &&
    e.status === "ok",
);
const measured = nativeReadback("first");
if (!first || !measured.receipt?.target_accepted)
  throw new Error("Turn 1 has no accepted host receipt: " + run.runId);
const secondRun = await Run.open(settings, run.runId);
register(secondRun, 2);
try {
  await secondRun.send(
    "You reconnected to the same session. Retry control_target_and_step with exactly the original arguments below, query control_receipt for the same command_id, then call control_stop and report the confirmed stopping step. Use only the attached MCP tools.\n" +
      JSON.stringify(first.params.arguments.arguments),
    opts,
  );
} finally {
  await secondRun.close();
}
const confirmed = entries().filter(
  (e) =>
    e.client?.run_id === run.runId && e.client?.turn === 2 && e.status === "ok",
);
const replay = confirmed.find(
  (e) => e.params?.arguments?.operation === "target_and_step",
);
const recovered = confirmed.find(
  (e) => e.params?.arguments?.operation === "receipt",
);
const stop = confirmed.find((e) => e.params?.arguments?.operation === "stop");
if (
  !replay ||
  !recovered ||
  !stop ||
  stop.result.control_step !== 1 ||
  stop.result.state !== "stopped" ||
  JSON.stringify(replay.result) !== JSON.stringify(first.result) ||
  JSON.stringify(recovered.result) !== JSON.stringify(first.result) ||
  first.client.pid === replay.client.pid
)
  throw new Error("Missing independent reconnect/stop receipt: " + run.runId);
const after = nativeReadback("after");
assert.deepEqual(
  after.robot_a.joint_target_position,
  before.robot_a.joint_target_position,
);
assert.deepEqual(after.receipt, measured.receipt);
assert.deepEqual(
  after.robot_b.joint_position,
  after.receipt.measured_after.qpos,
);
assert.ok(
  args.target.every(
    (v, i) => Math.abs(v - after.robot_b.joint_target_position[i]) < 1e-6,
  ),
);
assert.notDeepEqual(
  before.robot_b.joint_position,
  after.robot_b.joint_position,
);
assert.equal(after.receipt.physics_steps, 4);
if (before.robot_b.step_index !== null)
  assert.equal(after.robot_b.step_index - before.robot_b.step_index, 4);
fs.writeFileSync(
  output,
  JSON.stringify(
    {
      before,
      after,
      session_id: sessionId,
      run_id: run.runId,
      turns: [1, 2],
      client_pids: [first.client.pid, replay.client.pid],
      receipt: after.receipt,
      journal_results: {
        first: first.result,
        replay: replay.result,
        recovered: recovered.result,
      },
      replay_matches: true,
      stop: stop.result,
      journal: journalPath,
    },
    null,
    2,
  ),
);
console.log("Verified DexPilot Run", run.runId, "report", output);
