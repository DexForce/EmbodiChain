// ----------------------------------------------------------------------------
// Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ----------------------------------------------------------------------------

import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { delimiter } from "node:path";

export default function setup() {
  const repo = fileURLToPath(new URL("../../../", import.meta.url));
  const output = fileURLToPath(
    new URL("../.browser-fixtures", import.meta.url),
  );
  const args = [
    "tests/gen_sim/task_engine/browser_fixture.py",
    "--output",
    output,
  ];
  if (process.env.GRAPH_VIEWER_REFERENCE_HTML)
    args.push("--reference", process.env.GRAPH_VIEWER_REFERENCE_HTML);
  execFileSync(process.env.PYTHON || "python", args, {
    cwd: repo,
    env: {
      ...process.env,
      PYTHONPATH: [repo, process.env.PYTHONPATH]
        .filter(Boolean)
        .join(delimiter),
    },
    stdio: "inherit",
  });
}
