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

export interface Evidence {
  path: string;
  symbol: string;
  start_line: number;
  end_line: number;
  excerpt: string;
}
export interface ArchitectureNode {
  id: string;
  label: string;
  kind: string;
  topic_ids: string[];
  summary: string;
  boundaries: string[];
  evidence: Evidence[];
  documentation: { docname: string; label: string }[];
}
export interface ArchitectureEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  description: string;
  provenance: string;
  scope: string;
  evidence: Evidence[];
}
export interface ArchitectureView {
  id: string;
  label: string;
  description: string;
  node_ids: string[];
  edge_ids: string[];
  groups: { id: string; label: string; node_ids: string[] }[];
}
export interface ArchitectureSnapshot {
  schema_version: number;
  revision: string;
  source_ref: string;
  package_version: string;
  nodes: ArchitectureNode[];
  edges: ArchitectureEdge[];
  views: ArchitectureView[];
  limitations: string[];
}
export interface ExplorerState {
  viewId: string;
  nodeId: string | null;
  relations: string[];
  query: string;
}
