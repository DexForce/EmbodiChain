// Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { execFileSync, spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { setTimeout as delay } from "node:timers/promises";

const [dexPilotEntry, pythonArg, outputArg] = process.argv.slice(2);
if (!outputArg) {
  throw new Error(
    "Usage: node dexpilot_ur5_motion.mjs DEXPILOT_ENTRY PYTHON OUTPUT_DIR",
  );
}
const demo = path.dirname(fileURLToPath(import.meta.url));
const checkout = path.resolve(demo, "../../..");
const python = path.resolve(pythonArg);
const output = path.resolve(outputArg);
fs.mkdirSync(output, { mode: 0o700 });
const sessions = path.join(output, "sessions");
const receipts = path.join(output, "receipts");
const env = {
  ...process.env,
  PYTHONPATH: checkout,
  PYTHONDONTWRITEBYTECODE: "1",
};
const { Run, loadSettings } = await import(
  pathToFileURL(path.resolve(dexPilotEntry)).href
);
const settings = loadSettings();
settings.session_directory = sessions;
settings.runs_root = path.join(output, "dexpilot-runs");
settings.private_root = path.join(output, "dexpilot-state");
settings.world_restarts = 0;
settings.agent_python = path.dirname(path.dirname(python));
settings.agent_sdk = execFileSync(
  python,
  [
    "-c",
    "from pathlib import Path; import dexsim; print(Path(dexsim.__file__).parent.parent)",
  ],
  { cwd: checkout, env, encoding: "utf8" },
).trim();

const hostLog = fs.openSync(path.join(output, "world.log"), "w");
const host = spawn(
  python,
  [
    "-m",
    "examples.learning.policy_runs.ur5_motion_host",
    "--directory",
    sessions,
    "--output",
    receipts,
  ],
  { cwd: checkout, env, stdio: ["ignore", hostLog, hostLog] },
);
fs.closeSync(hostLog);
let hostError;
host.on("error", (error) => {
  hostError = error;
});
const hostDone = new Promise((resolve) => host.once("exit", resolve));

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

async function waitForWorld() {
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline) {
    if (hostError) throw hostError;
    if (host.exitCode !== null || host.signalCode !== null)
      throw new Error("UR5 host exited during startup");
    const record =
      fs.existsSync(sessions) &&
      fs.readdirSync(sessions).find((name) => /^dsws_.*\.json$/.test(name));
    if (record) return readJson(path.join(sessions, record));
    await delay(100);
  }
  throw new Error("UR5 World startup timed out");
}

const runId = `ur5-motion-${Date.now()}-${randomUUID().slice(0, 8)}`;
function mountMotion(turn) {
  settings.mcp_servers = [
    {
      name: "embodichain",
      command: python,
      args: [
        "-m",
        "embodichain",
        "mcp",
        "--session",
        "{session_id}",
        "--session-directory",
        "{session_directory}",
        "--motion-output",
        receipts,
        "--writer-key-file",
        path.join(sessions, "writer.key"),
        "--run-id",
        runId,
        "--turn",
        String(turn),
      ],
      env: [{ name: "PYTHONPATH", value: checkout }],
    },
  ];
}

const options = {
  onProgress: (line) => console.log(line),
  onPermission: ({ toolCall, options }) => {
    const title = String(toolCall.title ?? "");
    return title.includes("embodichain_")
      ? (options.find(({ kind }) => kind === "allow_once")?.optionId ?? null)
      : null;
  },
};

function receiptFor(command) {
  const matches = fs
    .readdirSync(receipts)
    .filter((name) => name.startsWith("ecmotion_"))
    .map((name) => readJson(path.join(receipts, name, "receipt.json")))
    .filter((item) => item.command_id === command);
  assert.equal(matches.length, 1);
  assert.equal(matches[0].status, "succeeded", JSON.stringify(matches[0]));
  return matches[0];
}

try {
  const world = await waitForWorld();
  mountMotion(1);
  const first = await Run.create(settings, {
    run_id: runId,
    task: [
      "使用配置好的 EmbodiChain MCP 让 UR5 单独抓起 cube 并保持抓持。",
      "先查 list_capabilities，然后调用 start_skill，skill_id=pick_up、object_ref=cube、command_id=demo-pick。",
      "用 get_run 等到 done 后给出 grasp_ref 和 run_id，并结束这一轮。回执由宿主自动保存。",
      `同一仿真 Session 为 ${world.session_id}。`,
    ].join("\n"),
    session_id: world.session_id,
    agent: "dsh",
    export_api: false,
    pack: path.join(demo, "motion_pack"),
    wall_seconds: 420,
    max_model_requests: 64,
  });
  const workOutput = path.join(first.root, "work", "output");
  fs.mkdirSync(workOutput, { recursive: true });
  fs.writeFileSync(
    path.join(workOutput, "recovery.py"),
    [
      '"""Rebuild the qualified fixture on an empty matching World."""',
      "from contextlib import ExitStack",
      "from examples.learning.policy_runs.ur5_motion_host import _DEPLOYMENT",
      "from examples.learning.policy_runs.ur5_pick_place import _Episode",
      "",
      "def run(context, **args):",
      "    if context.scene is None:",
      "        with ExitStack() as stack:",
      "            _Episode(context, _DEPLOYMENT, True, stack)",
      '    return {"session_id": context.session_id, "fixture_ready": True}',
      "",
    ].join("\n"),
  );
  try {
    const turn = await first.send(null, options);
    fs.writeFileSync(
      path.join(output, "turn-1.json"),
      JSON.stringify(turn, null, 2),
    );
    assert.equal(turn.end_reason, "replied", turn.error);
  } finally {
    await first.close();
  }
  const pick = receiptFor("demo-pick");
  assert.equal(pick.session_id, world.session_id);
  assert.ok(pick.result.verified && pick.result.lift_m >= 0.03);

  mountMotion(2);
  const second = Run.open(settings, runId);
  try {
    const turn = await second.send(
      [
        `查询第一轮的回执 ${pick.run_id}，并用 get_grasp 核对 ${pick.result.grasp_ref}。`,
        "拿到 verified 后，以 start_skill 提交 place，object_ref=cube、target_ref=platform_top、command_id=demo-place，携带该 grasp_ref。",
        "轮询 get_run 直到 done，直接回复最终验收结果和视频路径。宿主保存全部产物。",
      ].join("\n"),
      options,
    );
    fs.writeFileSync(
      path.join(output, "turn-2.json"),
      JSON.stringify(turn, null, 2),
    );
    assert.equal(turn.end_reason, "replied", turn.error);
  } finally {
    await second.close();
  }

  const place = receiptFor("demo-place");
  assert.equal(place.session_id, world.session_id);
  assert.ok(
    place.result.passed && Object.values(place.result.checks).every(Boolean),
  );
  assert.equal(place.done, true);
  const journal = fs
    .readFileSync(world.journal, "utf8")
    .trim()
    .split("\n")
    .map(JSON.parse);
  const calls = journal.filter(
    (item) =>
      item.params?.module === "embodichain.lab.sim.dexsim_session._motion_jobs",
  );
  const starts = calls.filter(
    (item) => item.params.arguments.operation === "start_skill",
  );
  assert.equal(starts.length, 2);
  assert.equal(starts[0].client.run_id, runId);
  assert.equal(starts[1].client.run_id, runId);
  assert.equal(starts[0].client.turn, 1);
  assert.equal(starts[1].client.turn, 2);
  assert.notEqual(starts[0].client.pid, starts[1].client.pid);
  const recheck = calls.find(
    (item) =>
      item.client.turn === 2 && item.params.arguments.operation === "get_grasp",
  );
  assert.ok(recheck && recheck.client.turn === 2 && recheck.result.verified);
  assert.equal(recheck.result.physics_step, pick.physics_step);
  const summary = {
    dexpilot_run_id: runId,
    session_id: world.session_id,
    runtime: world.runtime,
    pick,
    place,
    verified_after_reconnect: recheck.result,
    journal: world.journal,
  };
  fs.writeFileSync(
    path.join(output, "summary.json"),
    JSON.stringify(summary, null, 2),
  );
  console.log(JSON.stringify(summary, null, 2));
} finally {
  if (host.exitCode === null && !hostError) {
    host.kill("SIGTERM");
    await Promise.race([hostDone, delay(5000, null, { ref: false })]);
  }
}
