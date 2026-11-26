// Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
import { pathToFileURL } from "node:url";
import path from "node:path";
import { randomUUID } from "node:crypto";
const [python, root, entry] = process.argv.slice(2);
const { Run, loadSettings } = await import(
  pathToFileURL(path.resolve(entry)).href
);
const settings = loadSettings(),
  runId = "create-" + randomUUID();
settings.agent_sdk = null;
settings.world_restarts = 0;
settings.mcp_servers = [
  {
    name: "gensim",
    command: "/usr/bin/xvfb-run",
    args: [
      "-a",
      python,
      "-m",
      "embodichain",
      "mcp",
      "--experiment-output",
      root,
      "--experiment-agent",
      runId,
    ],
    env: [],
  },
];
const task =
  "Only use the GenSim MCP tools. Create an experimental run with agent_session=" +
  runId +
  " and command_id=initial. Inspect and validate its initial program, then submit attempt one. Inspect with wait_seconds=20 until a terminal receipt, read the measured error. Keep the frozen goal and evaluator. Revise only offset_x_m based on the measured error so the cube lands at the scene goal. Validate and submit exactly one fresh attempt, inspect with bounded waits until it finishes, and report both physical verdicts, attempts, bundles, measurements and trajectory paths. Do not use the shell or create a DexSim World yourself. The GenSim worker owns each attempt.";
const r = await Run.create(settings, {
  task,
  run_id: runId,
  session_id: null,
  export_api: false,
  agent: "dsh",
  max_model_requests: 48,
  wall_seconds: 650,
});
console.log(runId);
try {
  const result = await r.send(task, {
    onPermission: (request) =>
      request.toolCall?.title?.includes("gensim_")
        ? (request.options.find((x) => x.kind === "allow_once")?.optionId ??
          null)
        : null,
  });
  console.log(result.end_reason, result.reply);
} finally {
  await r.close();
}
