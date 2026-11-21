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

import { describe, it, expect } from "vitest";
import fixture from "../../../tests/gen_sim/task_engine/data/semantic_graph_contract.json";
import { createModel, layoutGraph } from "../src/model";
import { graphSvg } from "../src/svg";
import type { ViewerData } from "../src/types";

function example(): ViewerData {
  return {
    graph: structuredClone(fixture.graph),
    task_spec: structuredClone(fixture.task_spec),
    validation: { coverage: structuredClone(fixture.coverage) },
    report: null,
    call_evidence: {},
    task_spec_evaluation: null,
    identities: {
      graph_digest: "f".repeat(64),
      integration_fingerprint: fixture.graph.integration_fingerprint,
    },
    presentation: { assets: [] },
  } as ViewerData;
}

describe("recorded task presentation", () => {
  it.each([
    {
      name: "render before cutout",
      render: "data:image/png;base64,cmVuZGVy",
      cutout: "data:image/png;base64,Y3V0b3V0",
      kind: "render",
    },
    {
      name: "render without cutout",
      render: "data:image/png;base64,cmVuZGVy",
      cutout: null,
      kind: "render",
    },
    {
      name: "cutout fallback",
      render: null,
      cutout: "data:image/png;base64,Y3V0b3V0",
      kind: "cutout",
    },
    {
      name: "unsafe render fallback",
      render: "https://example.invalid/render.png",
      cutout: "data:image/png;base64,Y3V0b3V0",
      kind: "cutout",
    },
    { name: "missing images", render: null, cutout: null, kind: undefined },
  ])(
    "shows object previews in stages and calls: $name",
    ({ render, cutout, kind }) => {
      const data = example();
      data.presentation.assets = [
        { uid: "a", name: "Original can", render, cutout },
      ];
      const before = JSON.stringify(data),
        model = createModel(data, "all", "en");
      for (const mode of ["groups", "calls"] as const) {
        const layout = layoutGraph(data, model, mode, "", "all", "overview");
        const row = layout.rows[0];
        expect(row.thumbnail).toBe(
          kind === "render" ? render : kind === "cutout" ? cutout : undefined,
        );
        expect(row.thumbnailKind).toBe(kind);
        expect(row.height).toBe(kind ? 248 : 148);
        if (kind)
          expect(graphSvg(layout, "", "", "licenses")).toContain(
            'data-image-kind="' + kind + '"',
          );
      }
      expect(JSON.stringify(data)).toBe(before);
    },
  );
  it("keeps independent image cards from overlapping in one channel", () => {
    const data = example();
    data.presentation.assets = [
      { uid: "a", render: "data:image/png;base64,cmVuZGVy" },
    ];
    data.graph.nodes = data.graph.nodes.slice(0, 2);
    data.graph.nodes[1].depends_on = [];
    data.graph.task_groups[0].node_ids = data.graph.nodes.map(
      (node) => node.id,
    );
    const layout = layoutGraph(
      data,
      createModel(data, "all", "en"),
      "calls",
      "",
      "all",
      "overview",
    );
    expect(layout.rows[0].y + layout.rows[0].height).toBeLessThan(
      layout.rows[1].y,
    );
  });
  it("keeps missing observations unknown and never creates a certificate", () => {
    const data = example(),
      before = JSON.stringify(data),
      model = createModel(data, "all", "en");
    expect(model.callStatus(data.graph.nodes[0])).toBe("unknown");
    expect(model.groupStatus(data.graph.task_groups[0])).toBe("unknown");
    expect(model.goalStatus(data.task_spec!.goal[0])).toBe("unknown");
    expect(model.strict).toBe("unavailable");
    expect(JSON.stringify(data)).toBe(before);
  });
  it("preserves per-environment participation and separates stage/program/normative results", () => {
    const data = example(),
      node = data.graph.nodes[0],
      group = data.graph.task_groups[0],
      goal = data.task_spec!.goal[0];
    data.report = {
      status: "succeeded",
      record_dir: "run",
      environments: [
        {
          env_id: 0,
          success: true,
          terminal_reason: "original",
          semantic_success: { [group.id]: true },
        },
        {
          env_id: 1,
          success: false,
          terminal_reason: "original",
          semantic_success: { [group.id]: false },
        },
      ],
      runtime_result: { success: [true, true] },
    };
    data.call_evidence[node.id] = {
      name: node.id,
      active: [true, false],
      successes: [true, true],
      metadata: { runtime: { env_ids: [0, 1] } },
    };
    data.task_spec_evaluation = {
      environments: [
        {
          env_id: 0,
          status: "satisfied",
          certificate_ready: true,
          goal: [{ predicate_id: goal.id, status: "satisfied" }],
        },
        {
          env_id: 1,
          status: "unavailable",
          certificate_ready: false,
          goal: [{ predicate_id: goal.id, status: "unavailable" }],
        },
      ],
    };
    const all = createModel(data, "all", "en"),
      first = createModel(data, "0", "en"),
      second = createModel(data, "1", "en");
    expect(all.callStatus(node)).toBe("unknown");
    expect(first.callStatus(node)).toBe("succeeded");
    expect(second.callStatus(node)).toBe("unknown");
    expect(all.groupStatus(group)).toBe("failed");
    expect(all.goalStatus(goal)).toBe("unavailable");
    expect(all.strict).toBe("unavailable");
    expect(first.strict).toBe("satisfied");
    expect(second.strict).toBe("unavailable");
    data.call_evidence[node.id].metadata!.runtime!.env_ids = [0, 0];
    expect(createModel(data, "0", "en").callStatus(node)).toBe("unknown");
  });
  it("retains dependency topology, logical channels and raw identities in vector export", () => {
    const data = example();
    const model = createModel(data, "all", "en");
    const layout = layoutGraph(data, model, "calls", "", "all", "overview");
    expect(layout.rows.map((row) => row.id)).toEqual(
      data.graph.nodes.map((node) => node.id),
    );
    expect(layout.edges.length).toBe(
      data.graph.nodes.reduce(
        (count, node) => count + node.depends_on.length,
        0,
      ),
    );
    for (const edge of layout.edges)
      expect(
        layout.rows.find((row) => row.key === edge.source)!.x,
      ).toBeLessThan(layout.rows.find((row) => row.key === edge.target)!.x);
    layout.rows[0].title = "<script>literal & label</script>";
    const svg = graphSvg(layout, "", "", "original licenses");
    expect(svg).toContain("&lt;script&gt;literal &amp; label&lt;/script&gt;");
    expect(svg).not.toContain("<script>");
    for (const node of data.graph.nodes)
      expect(svg).toContain('data-key="call:' + node.id + '"');
    expect(svg).toContain("original licenses");
  });
  it("names the dual-arm rail explicitly instead of reporting it as unbound", () => {
    const data = example();
    data.graph.nodes[0].call.resources = {
      source: "left",
      destination: "right",
    };
    const group = data.graph.task_groups[0];
    group.node_ids = [data.graph.nodes[0].id];
    data.graph.nodes = [data.graph.nodes[0]];
    data.task_spec = null;
    data.validation = null;
    const layout = layoutGraph(
      data,
      createModel(data, "all", "zh"),
      "groups",
      "",
      "all",
      "overview",
    );
    expect(layout.rows[0].control).toBe("coordinated");
    expect(layout.lanes[0].label).toBe("双臂协同");
  });
});
