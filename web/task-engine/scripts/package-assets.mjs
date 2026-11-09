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

import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  readFileSync,
  writeFileSync,
  readdirSync,
  existsSync,
  renameSync,
} from "node:fs";
const app = fileURLToPath(new URL("..", import.meta.url));
const repo = resolve(app, "../..");
const seen = new Set(),
  notices = [];
function locate(name, from) {
  const require = createRequire(join(from, "package.json"));
  let entry;
  try {
    entry = require.resolve(name + "/package.json");
  } catch {
    entry = require.resolve(name);
  }
  let root = dirname(entry);
  while (root !== dirname(root)) {
    const file = join(root, "package.json");
    if (
      existsSync(file) &&
      JSON.parse(readFileSync(file, "utf8")).name === name
    )
      return root;
    root = dirname(root);
  }
  throw new Error("Cannot locate installed dependency: " + name);
}
function collect(name, from) {
  const root = locate(name, from),
    pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));
  const key = pkg.name + "@" + pkg.version;
  if (seen.has(key)) return;
  seen.add(key);
  if (!pkg.name.startsWith("@embodichain/")) {
    const licenses = readdirSync(root).filter((name) =>
      /^(licen[cs]e|copying|ofl)([.\-_]|$)/i.test(name),
    );
    if (!licenses.length) throw new Error("Dependency license missing: " + key);
    notices.push(
      key +
        "\n" +
        licenses
          .map((name) => readFileSync(join(root, name), "utf8"))
          .join("\n"),
    );
  }
  for (const dependency of Object.keys(pkg.dependencies || {}))
    collect(dependency, root);
}
const pkg = JSON.parse(readFileSync(join(app, "package.json"), "utf8"));
for (const name of Object.keys(pkg.dependencies)) collect(name, app);
const destination = join(repo, "embodichain/gen_sim/task_engine");
const files = {
  "_graph_react.js": readFileSync(
    join(app, "dist/_graph_react.js"),
    "utf8",
  ).replace(/<\/script/gi, "<\\/script"),
  "_graph_react.css": readFileSync(join(app, "dist/_graph_react.css"), "utf8"),
  "_graph_react.LICENSE.txt": notices
    .sort()
    .join("\n\n" + "=".repeat(72) + "\n\n"),
};
for (const [name, value] of Object.entries(files)) {
  const temporary = join(destination, "." + name + ".tmp");
  writeFileSync(temporary, value);
  renameSync(temporary, join(destination, name));
}
console.log(
  "Packaged offline graph assets and " +
    notices.length +
    " dependency notices.",
);
