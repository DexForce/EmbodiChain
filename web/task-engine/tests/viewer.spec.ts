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

import { expect, test, type Page } from "@playwright/test";
import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join } from "node:path";
import type { ViewerData } from "../src/types";

const root = fileURLToPath(new URL("../.browser-fixtures", import.meta.url));
const cases = [
  "static-absent",
  "static-spec",
  "passed",
  "failed",
  "mixed",
  "unavailable",
  "aborted",
  "rejected",
  "literal",
];
type Snapshot = {
  calls: Record<string, string>;
  groups: Record<string, string>;
  goals: Record<string, string>;
};
type Fixture = {
  expected: Record<string, Snapshot & { strict: string }>;
  reference: boolean;
  instruction: string;
};
type Bridge = {
  data: ViewerData;
  selected: () => string;
  mode: () => string;
  callStatus: (node: ViewerData["graph"]["nodes"][number]) => string;
  groupStatus: (group: ViewerData["graph"]["task_groups"][number]) => string;
  goalStatus: (
    goal: NonNullable<ViewerData["task_spec"]>["goal"][number],
  ) => string;
};
declare global {
  interface Window {
    __graphViewer: Bridge;
    __injection?: boolean;
  }
}
const fixture = (name: string): Fixture =>
  JSON.parse(readFileSync(join(root, "manifest.json"), "utf8")).cases[name];
const data = (name: string): ViewerData =>
  JSON.parse(readFileSync(join(root, name + ".json"), "utf8"));
const label: Record<string, string> = {
  succeeded: "Passed",
  satisfied: "Satisfied",
  failed: "Failed",
  unknown: "Unverified",
  unavailable: "Evidence unavailable",
};

async function open(page: Page, name: string, reference = false, query = "") {
  await page.goto(
    pathToFileURL(join(root, (reference ? "reference-" : "") + name + ".html"))
      .href + query,
  );
  await page.waitForFunction(() => Boolean(window.__graphViewer));
  await page.evaluate(() => document.fonts.ready);
}
async function snapshot(page: Page): Promise<Snapshot> {
  return page.evaluate(() => {
    const v = window.__graphViewer,
      d = v.data;
    return {
      calls: Object.fromEntries(
        d.graph.nodes.map((n) => [n.id, v.callStatus(n)]),
      ),
      groups: Object.fromEntries(
        d.graph.task_groups.map((g) => [g.id, v.groupStatus(g)]),
      ),
      goals: Object.fromEntries(
        (d.task_spec?.goal || []).map((g) => [g.id, v.goalStatus(g)]),
      ),
    };
  });
}
async function exported(page: Page, id: string) {
  const download = page.waitForEvent("download");
  await page.locator(id).click();
  const path = await (await download).path();
  return readFileSync(path!, "utf8");
}

test.beforeEach(async ({ context, page }) => {
  await context.route(/^https?:\/\//, (route) =>
    route.abort("internetdisconnected"),
  );
  await context.addInitScript(() => {
    try {
      if (!localStorage.getItem("gensim.graph.language"))
        localStorage.setItem("gensim.graph.language", "en");
    } catch {
      /* Denied storage is supported. */
    }
  });
  const errors: string[] = [],
    external: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => {
    if (/^https?:/.test(request.url())) external.push(request.url());
  });
  (page as Page & { diagnostics: unknown }).diagnostics = { errors, external };
});
test.afterEach(async ({ page }, info) => {
  const diagnostics = (
    page as Page & { diagnostics: { errors: string[]; external: string[] } }
  ).diagnostics;
  await info.attach("offline-diagnostics", {
    body: JSON.stringify(diagnostics),
    contentType: "application/json",
  });
  expect(diagnostics.errors).toEqual([]);
  expect(diagnostics.external).toEqual([]);
});

for (const name of cases)
  test(`${name}: visible states retain recorded facts and reference behavior`, async ({
    page,
  }) => {
    const entry = fixture(name),
      input = data(name);
    const baseline: Record<string, Snapshot> = {};
    if (entry.reference) {
      await open(page, name, true);
      for (const environment of Object.keys(entry.expected)) {
        if (input.report)
          await page.locator("#environment").selectOption(environment);
        baseline[environment] = await snapshot(page);
      }
    }
    await open(page, name);
    for (const [environment, expected] of Object.entries(entry.expected)) {
      if (input.report)
        await page.locator("#environment").selectOption(environment);
      const { strict, ...states } = expected;
      await expect.poll(() => snapshot(page)).toEqual(states);
      if (entry.reference)
        expect(await snapshot(page)).toEqual(baseline[environment]);
      await page.locator("#mode-calls").click();
      await expect(page.locator(".task-graph-node")).toHaveCount(
        input.graph.nodes.length,
      );
      for (const [id, status] of Object.entries(expected.calls)) {
        const card = page.locator(`.task-graph-node[data-key="call:${id}"]`);
        await expect(card).toHaveAttribute("data-status", status);
        await expect(card.locator(".task-node-status")).toHaveText(
          label[status],
        );
      }
      await page.locator("#mode-groups").click();
      for (const [id, status] of Object.entries(expected.groups))
        await expect(
          page.locator(`.task-graph-node[data-key="group:${id}"]`),
        ).toHaveAttribute("data-status", status);
      await expect(page.locator(".task-graph-node .task-node-sub")).toHaveText(
        `6 Calls / ${input.report ? "60" : "Not recorded"} steps`,
      );
      for (const [id, status] of Object.entries(expected.goals)) {
        await expect(
          page
            .locator(".goal-card")
            .filter({ hasText: id })
            .locator(".status-badge"),
        ).toHaveText(label[status]);
        await expect(
          page
            .locator(".goal-card")
            .filter({ hasText: id })
            .locator(":scope > svg"),
        ).toHaveClass(
          status === "failed"
            ? /lucide-circle-x/
            : status === "satisfied"
              ? /lucide-circle-check/
              : /lucide-circle-alert/,
        );
      }
      await expect(page.locator(".summary-status")).toContainText(
        strict === "satisfied"
          ? "Strict acceptance passed"
          : strict === "failed"
            ? "Strict acceptance failed"
            : "Strict evidence incomplete",
      );
    }
    if (name === "static-absent") {
      await expect(page.locator("#environment")).toBeDisabled();
      await expect(page.locator(".task-goals")).toContainText(
        "TaskSpec not provided",
      );
    }
    if (name === "literal") {
      await expect(page.locator(".task-summary p")).toHaveText(
        entry.instruction,
      );
      expect(await page.evaluate(() => window.__injection)).toBeUndefined();
    }
    if (name === "literal" || name === "rejected")
      await expect(page.locator("video")).toHaveCount(0);
    expect(await exported(page, "#export-json")).toBe(
      JSON.stringify(input, null, 2),
    );
  });

test("selection, goal highlighting, raw data and search/status/reset controls", async ({
  page,
}) => {
  await open(page, "mixed");
  await page.locator("#mode-calls").click();
  const card = page.locator(
    '.task-graph-node[data-key="call:step_01__call_01"]',
  );
  await card.focus();
  await page.keyboard.press("Space");
  await page.keyboard.press("Backspace");
  await expect(page.locator(".task-graph-node")).toHaveCount(6);
  await expect(page.locator("#detail .sub-id")).toHaveText("step_01__call_01");
  await expect(page.locator(".event-list")).toContainText("action_completed");
  await page.locator(".goal-card").click();
  await expect(page.locator("#detail")).toContainText("goal_001");
  await page.locator("#search").fill("step_01__call_01");
  await expect(page.locator(".task-graph-node.faded")).toHaveCount(4);
  await page.locator("#search").fill("");
  await page.locator("#status-filter").selectOption("failed");
  await expect(page.locator(".task-graph-node.faded")).toHaveCount(5);
  await page.locator("#environment").selectOption("7");
  await expect(page.locator(".task-graph-node.faded")).toHaveCount(6);
  await page.locator("#reset").click();
  await expect(page.locator("#search")).toHaveValue("");
  await expect(page.locator("#status-filter")).toHaveValue("all");
  await expect(page.locator("#mode-groups")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator(".reading-size")).toHaveText("100%");
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await expect
    .poll(async () => parseInt(await page.locator(".reading-size").innerText()))
    .toBeGreaterThan(110);
  await page.locator("#reset").click();
  await expect(page.locator(".reading-size")).toHaveText("100%");
  await page.locator("#provenance").click();
  const graph = page
    .locator("#detail details")
    .filter({ has: page.locator("summary", { hasText: "Original graph" }) });
  await graph.locator("summary").click();
  expect(JSON.parse(await graph.locator("pre").innerText())).toEqual(
    data("mixed").graph,
  );
});

test("object renders are visible inside stage and call cards and SVG exports", async ({
  page,
}) => {
  for (const [name, kind] of [
    ["passed", "render"],
    ["unavailable", "render"],
    ["aborted", "cutout"],
  ] as const) {
    await open(page, name);
    const input = data(name),
      source = input.presentation.assets.find((asset) => asset.uid === "a")![
        kind
      ];
    for (const mode of ["groups", "calls"] as const) {
      await page.locator("#mode-" + mode).click();
      const images = page.locator(".task-graph-node .task-node-image");
      await expect(images).toHaveCount(mode === "groups" ? 1 : 5);
      for (const image of await images.all()) {
        await expect(image).toHaveAttribute("src", source!);
        await expect(image).toHaveAttribute("data-image-kind", kind);
        await expect
          .poll(() =>
            image.evaluate(
              (element: HTMLImageElement) =>
                element.complete && element.naturalWidth > 0,
            ),
          )
          .toBe(true);
      }
      const svg = await exported(page, "#export-svg");
      const exportedImages = await page.evaluate(
        (text) =>
          [
            ...new DOMParser()
              .parseFromString(text, "image/svg+xml")
              .querySelectorAll("image"),
          ].map((image) => image.getAttribute("href")),
        svg,
      );
      expect(exportedImages).toEqual(
        Array(mode === "groups" ? 1 : 5).fill(source),
      );
    }
  }
});

test("asset metadata, cutout fallback and modal remain available", async ({
  page,
}) => {
  await open(page, "passed");
  const can = page.locator(".asset-row").filter({ hasText: "Display can" });
  await can.locator("button").last().click();
  await expect(page.locator("#detail")).toContainText("Original can");
  await expect(page.locator("#detail")).toContainText("assets/a/model.glb");
  await expect(page.locator("#detail")).toContainText("a".repeat(64));
  await page.locator("#detail .inspector-image").click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByRole("button", { name: "Close", exact: true }),
  ).toBeFocused();
  const render = await dialog.locator("img").getAttribute("src");
  await dialog
    .getByRole("button", { name: "Source cutout", exact: true })
    .click();
  expect(await dialog.locator("img").getAttribute("src")).not.toBe(render);
  await expect(dialog).toContainText("assets/a/cutout.png");
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(page.locator("#inspector")).toBeVisible();
  await expect(page.locator(".asset-preview-trigger")).toBeFocused();
  const table = page.locator(".asset-row").filter({ hasText: "Cutout only" });
  await table.locator("button").last().click();
  await expect(page.locator("#detail .inspector-image")).toBeVisible();
  await table.locator(".asset-thumb").click();
  await expect(
    dialog.getByRole("button", { name: "GLB render", exact: true }),
  ).toBeDisabled();
  await expect(dialog.locator("img")).toBeVisible();
  await dialog.getByRole("button", { name: "Close", exact: true }).click();
  await page
    .locator(".asset-row")
    .filter({ hasText: "Missing preview" })
    .locator("button")
    .last()
    .click();
  await expect(page.locator("#detail")).toContainText(
    "Asset image unavailable",
  );
  expect(await page.locator('.asset-row img[src^="http"]').count()).toBe(0);
});

test("local video metadata, playback and original-file link work offline", async ({
  page,
}) => {
  test.skip(
    !JSON.parse(readFileSync(join(root, "manifest.json"), "utf8")).video,
    "ffmpeg is required to create the local media fixture",
  );
  await open(page, "passed");
  const video = page.locator("video");
  await expect
    .poll(() => video.evaluate((node: HTMLVideoElement) => node.readyState))
    .toBeGreaterThanOrEqual(2);
  expect(
    await video.evaluate((node: HTMLVideoElement) => node.duration),
  ).toBeCloseTo(1, 1);
  await video.evaluate((node: HTMLVideoElement) => node.play());
  await expect
    .poll(() => video.evaluate((node: HTMLVideoElement) => node.currentTime))
    .toBeGreaterThan(0);
  await video.evaluate((node: HTMLVideoElement) => node.pause());
  await expect(
    page.getByRole("link", { name: "Open original video" }),
  ).toHaveAttribute("href", "videos/env_7/front%20end.mp4");
});

test("language, original instruction, denied storage and dark theme", async ({
  page,
  context,
}) => {
  await open(page, "passed", false, "?theme=dark");
  await expect(page.locator(".task-app")).toHaveAttribute("data-theme", "dark");
  await page.locator('[data-language="zh"]').click();
  await expect(page.locator("html")).toHaveAttribute("lang", "zh-CN");
  await expect(page.locator("h1")).toHaveText("前端展示验证");
  await page.locator(".original-instruction summary").click();
  await expect(page.locator(".original-instruction")).toContainText(
    data("passed").graph.instruction,
  );
  await page.getByRole("button", { name: "切换主题", exact: true }).click();
  await expect(page.locator(".task-app")).toHaveAttribute(
    "data-theme",
    "light",
  );
  await page.reload();
  await expect(page.locator("h1")).toHaveText("前端展示验证");
  await context.addInitScript(() =>
    Object.defineProperty(window, "localStorage", {
      get() {
        throw new Error("Storage is unavailable");
      },
    }),
  );
  await page.reload();
  await page.waitForFunction(() => Boolean(window.__graphViewer));
  await page.locator('[data-language="en"]').click();
  await expect(page.locator("h1")).toHaveText("Frontend display validation");
  expect(JSON.parse(await exported(page, "#export-json"))).toEqual(
    data("passed"),
  );
});

test("zoom, pan, fit and vector export preserve all offscreen dependencies", async ({
  page,
}) => {
  await open(page, "passed");
  await page.locator("#mode-calls").click();
  const percent = page.locator(".reading-size");
  await page
    .getByRole("button", { name: "Read at 100 percent", exact: true })
    .click();
  await expect(percent).toHaveText("100%");
  await page.getByRole("button", { name: "Zoom in", exact: true }).click();
  await expect
    .poll(async () => parseInt(await percent.innerText()))
    .toBeGreaterThan(110);
  const enlarged = parseInt(await percent.innerText());
  await page.getByRole("button", { name: "Zoom out", exact: true }).click();
  await expect
    .poll(async () => parseInt(await percent.innerText()))
    .toBeLessThan(enlarged);
  await page
    .getByRole("button", { name: "Read at 100 percent", exact: true })
    .click();
  await expect(percent).toHaveText("100%");
  const viewport = page.locator(".react-flow__viewport"),
    before = await viewport.getAttribute("style"),
    bounds = (await page.locator(".task-canvas").boundingBox())!;
  await page.mouse.move(bounds.x + 30, bounds.y + 180);
  await page.mouse.down();
  await page.mouse.move(bounds.x + 100, bounds.y + 230, { steps: 5 });
  await page.mouse.up();
  await expect(viewport).not.toHaveAttribute("style", before!);
  await page.getByRole("button", { name: "Fit graph", exact: true }).click();
  for (const mode of ["groups", "calls"]) {
    await page.locator("#mode-" + mode).click();
    await page
      .getByRole("button", { name: "Toggle theme", exact: true })
      .click();
    const svg = await exported(page, "#export-svg");
    const parsed = await page.evaluate((text) => {
      const svg = new DOMParser().parseFromString(text, "image/svg+xml");
      return {
        error: svg.querySelector("parsererror")?.textContent,
        keys: [...svg.querySelectorAll("[data-key]")].map((n) =>
          n.getAttribute("data-key"),
        ),
        edges: [...svg.querySelectorAll("[data-from]")].map((n) => [
          n.getAttribute("data-from"),
          n.getAttribute("data-to"),
        ]),
        rows: Object.fromEntries(
          [...svg.querySelectorAll("[data-key]")].map((n) => [
            n.getAttribute("data-key"),
            Number(
              n.getAttribute("transform")?.match(/translate\(([^,]+)/)?.[1],
            ),
          ]),
        ),
        background: svg.documentElement
          .querySelector(":scope > rect")
          ?.getAttribute("fill"),
        theme: svg.documentElement.getAttribute("data-theme"),
        fonts: svg.querySelector("style")?.textContent,
        metadata: svg.querySelector("metadata")?.textContent,
      };
    }, svg);
    expect(parsed.error).toBeUndefined();
    const definitions =
      mode === "groups"
        ? data("passed").graph.task_groups
        : data("passed").graph.nodes;
    const prefix = mode === "groups" ? "group:" : "call:";
    expect(parsed.keys.sort()).toEqual(
      definitions.map((row) => prefix + row.id).sort(),
    );
    expect(parsed.edges).toHaveLength(
      definitions.reduce((n, row) => n + row.depends_on.length, 0),
    );
    for (const row of definitions)
      for (const dependency of row.depends_on)
        expect(parsed.rows[prefix + dependency]).toBeLessThan(
          parsed.rows[prefix + row.id],
        );
    expect(parsed.theme).toBe(
      await page.locator(".task-app").getAttribute("data-theme"),
    );
    expect(parsed.fonts).toContain("data:font/");
    expect(parsed.metadata).toBe(
      JSON.parse((await page.locator("#font-license-data").textContent())!),
    );
    expect(parsed.background).toBe(
      await page
        .locator(".task-app")
        .evaluate((node) =>
          getComputedStyle(node).getPropertyValue("--canvas").trim(),
        ),
    );
  }
});

for (const width of [390, 834])
  test(`viewport ${width}: all navigation and media remain reachable`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1000 });
    await open(page, "passed");
    await expect(page.locator("#inspector")).toHaveCount(0);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth),
    ).toBeLessThanOrEqual(width + 1);
    await page.getByRole("button", { name: "Assets", exact: true }).click();
    await page
      .locator(".asset-row")
      .filter({ hasText: "Display can" })
      .locator("button")
      .last()
      .click();
    await expect(page.locator("#detail .inspector-image")).toBeVisible();
    await page.getByRole("button", { name: "Close", exact: true }).click();
    await page.locator("#export-json").click();
    await page.locator("#provenance").click();
    await expect(page.locator("#detail")).toContainText("RUN / IDENTITY");
    await page.getByRole("button", { name: "Close", exact: true }).click();
    for (const resizedWidth of [1400, width]) {
      await page.setViewportSize({ width: resizedWidth, height: 1000 });
      await expect(page.locator(".task-canvas")).toHaveAttribute(
        "aria-busy",
        "false",
      );
      await expect
        .poll(async () => {
          const canvas = (await page.locator(".task-canvas").boundingBox())!;
          const node = (await page.locator(".task-graph-node").boundingBox())!;
          return (
            node.x >= canvas.x && node.x + node.width <= canvas.x + canvas.width
          );
        })
        .toBe(true);
    }
    await page.screenshot({
      path: join(root, `viewport-${width}.png`),
      fullPage: true,
    });
  });
