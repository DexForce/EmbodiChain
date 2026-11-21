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

import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  testMatch: "viewer.spec.ts",
  globalSetup: "./tests/browser.setup.ts",
  workers: 2,
  fullyParallel: true,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  use: {
    launchOptions: { env: { ...process.env, LIBGL_ALWAYS_SOFTWARE: "1" } },
    viewport: { width: 1680, height: 1120 },
    offline: true,
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        launchOptions: { args: ["--no-sandbox", "--disable-gpu"] },
      },
    },
    { name: "firefox", use: { browserName: "firefox" } },
    // WebKit's offline emulation rejects even plain file: navigations. HTTP(S)
    // is instead blocked by the same request guard applied to every project.
    { name: "webkit", use: { browserName: "webkit", offline: false } },
  ],
  reporter: [
    ["list"],
    ["json", { outputFile: "test-results/browser-results.json" }],
  ],
});
