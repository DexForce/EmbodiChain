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

export type Status =
  | "succeeded"
  | "satisfied"
  | "failed"
  | "unavailable"
  | "unknown"
  | "not_run"
  | "rejected"
  | "aborted";
export type Mode = "groups" | "calls";
export type Track = "left" | "auto" | "right";
export type Control = Track | "coordinated";
export interface Call {
  kind: string;
  call_id?: string;
  object?: string;
  arguments?: {
    object?: string;
    reference?: string;
    relation?: string;
    [key: string]: unknown;
  };
  resources: Record<string, string>;
  [key: string]: unknown;
}
export interface TaskNode {
  id: string;
  call: Call;
  depends_on: string[];
  task_instance_id: string;
  task_type: string;
  role: string;
}
export interface TaskGroup {
  id: string;
  task_type: string;
  node_ids: string[];
  depends_on: string[];
  success: unknown;
}
export interface Predicate {
  id: string;
  predicate: string;
  subject?: string;
  reference?: string;
  relation?: string;
  arguments?: Record<string, unknown>;
  [key: string]: unknown;
}
export interface RuntimeEvent {
  timestamp?: number;
  kind: string;
  segment_name?: string;
  [key: string]: unknown;
}
export interface Segment {
  name: string;
  start_step?: number;
  end_step?: number;
  active?: boolean[];
  successes?: boolean[];
  metadata?: {
    runtime?: {
      env_ids?: number[];
      events?: RuntimeEvent[];
      [key: string]: unknown;
    };
    [key: string]: unknown;
  };
  [key: string]: unknown;
}
export interface Environment {
  env_id: number;
  success: boolean;
  terminal_reason: string;
  semantic_success: Record<string, boolean>;
}
export interface GoalEvidence {
  predicate_id: string;
  status: Status;
  reason?: string;
  [key: string]: unknown;
}
export interface NormativeRow {
  env_id: number;
  status: Status;
  certificate_ready: boolean;
  goal: GoalEvidence[];
  [key: string]: unknown;
}
export interface Asset {
  uid: string;
  name?: string;
  description?: string;
  category?: string;
  role?: string;
  cutout?: string | null;
  render?: string | null;
  cutout_source?: string | null;
  render_source?: string | null;
  glb_source?: string;
  glb_sha256?: string;
  [key: string]: unknown;
}
export interface ViewerData {
  graph: {
    schema_version: string;
    task_id: string;
    instruction: string;
    integration_fingerprint: string;
    nodes: TaskNode[];
    task_groups: TaskGroup[];
    [key: string]: unknown;
  };
  task_spec: { goal: Predicate[]; [key: string]: unknown } | null;
  validation: {
    coverage: {
      goals: { id: string; groups: string[] }[];
      [key: string]: unknown;
    };
    [key: string]: unknown;
  } | null;
  report: {
    status: Status;
    record_dir: string;
    environments: Environment[];
    runtime_result: {
      length?: number;
      lengths?: number[];
      success?: boolean[];
      acceptance_mode?: string;
      [key: string]: unknown;
    } | null;
    [key: string]: unknown;
  } | null;
  call_evidence: Record<string, Segment>;
  task_spec_evaluation: {
    environments: NormativeRow[];
    [key: string]: unknown;
  } | null;
  identities: { graph_digest: string; integration_fingerprint: string };
  presentation: {
    assets: Asset[];
    run_id?: string;
    robot?: string;
    created_at?: string;
    wall_seconds?: number | null;
    wall_source?: string;
    acceptance_mode?: string;
    video_href?: string | null;
    localizations?: {
      title?: Record<string, string>;
      instruction?: Record<string, string>;
      assets?: Record<
        string,
        { name?: Record<string, string>; description?: Record<string, string> }
      >;
    };
    [key: string]: unknown;
  };
}
export interface ViewRow {
  key: string;
  id: string;
  title: string;
  index: string;
  subtitle: string;
  status: Status;
  thumbnail?: string;
  thumbnailKind?: "render" | "cutout";
  thumbnailLabel?: string;
  track: Track;
  control: Control;
  x: number;
  y: number;
  width: number;
  height: number;
  nodeIds: string[];
  faded: boolean;
}
export interface Lane {
  track: Track;
  label: string;
  y: number;
  height: number;
  width: number;
}
export interface Layout {
  rows: ViewRow[];
  lanes: Lane[];
  edges: { id: string; source: string; target: string }[];
  width: number;
  height: number;
}
