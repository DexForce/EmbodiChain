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

import { good } from "./model";
import type { Layout } from "./types";

export interface SvgPalette {
  name: "light" | "dark";
  canvas: string;
  panel: string;
  rail: string;
  text: string;
  muted: string;
  line: string;
  accent: string;
  good: string;
  bad: string;
  unknown: string;
}
const light: SvgPalette = {
  name: "light",
  canvas: "#faf9f6",
  panel: "#fdfcf9",
  rail: "#eaf0f5",
  text: "#263340",
  muted: "#596574",
  line: "#d7dadb",
  accent: "#375b7b",
  good: "#28715a",
  bad: "#b44a45",
  unknown: "#a77629",
};

const xml = (value: unknown) =>
  String(value)
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, "�")
    .replace(
      /[&<>"']/g,
      (character) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&apos;",
        })[character]!,
    );
/** Export the current data/layout as vector SVG; no DOM screenshot or network. */
export function graphSvg(
  layout: Layout,
  selected: string,
  styles: string,
  licenses: string,
  statusLabel: (status: string) => string = (value) => value,
  palette: SvgPalette = light,
): string {
  const fonts = (styles.match(/@font-face\s*\{[^}]*\}/g) || []).join("\n");
  const points = new Map(layout.rows.map((row) => [row.key, row]));
  const lanes = layout.lanes
    .map(
      (lane) =>
        `<g><rect x="12" y="${lane.y}" width="${lane.width}" height="${lane.height}" rx="9" fill="${xml(palette.rail)}"/><text x="26" y="${lane.y + 25}" fill="${xml(palette.muted)}" font-size="11">${xml(lane.label)}</text></g>`,
    )
    .join("");
  const edges = layout.edges
    .map((edge) => {
      const a = points.get(edge.source)!,
        b = points.get(edge.target)!,
        x = a.x + a.width,
        y = a.y + a.height / 2,
        end = b.y + b.height / 2,
        middle = (x + b.x) / 2;
      return `<path data-from="${xml(edge.source)}" data-to="${xml(edge.target)}" d="M${x} ${y} C${middle} ${y} ${middle} ${end} ${b.x - 4} ${end}" fill="none" stroke="#8b9aad" stroke-width="1.4" marker-end="url(#arrow)"/>`;
    })
    .join("");
  const rows = layout.rows
    .map((row) => {
      const color = good(row.status)
        ? palette.good
        : row.status === "failed"
          ? palette.bad
          : palette.unknown;
      return `<g data-key="${xml(row.key)}" opacity="${row.faded ? ".28" : "1"}" transform="translate(${row.x},${row.y})"><title>${xml(row.id)}</title><rect width="${row.width}" height="${row.height}" rx="8" fill="${xml(palette.panel)}" stroke="${xml(selected === row.key ? palette.accent : palette.line)}" stroke-width="${selected === row.key ? 2 : 1}"/>${row.thumbnail ? `<image href="${xml(row.thumbnail)}" x="16" y="98" width="${row.width - 32}" height="92" preserveAspectRatio="xMidYMid meet" data-image-kind="${xml(row.thumbnailKind || "render")}" aria-label="${xml(row.thumbnailLabel || row.title)}"/>` : ""}<text x="16" y="26" font-size="10" fill="${xml(palette.muted)}">${xml(row.index)}</text><svg x="16" y="40" width="${row.width - 30}" height="30"><text y="19" font-size="15" fill="${xml(palette.text)}">${xml(row.title)}</text></svg><text x="16" y="${row.thumbnail ? 85 : 94}" font-size="11" fill="${xml(palette.muted)}">${xml(row.subtitle)}</text><path d="M16 ${row.height - 38} H${row.width - 16}" stroke="${xml(palette.line)}"/><text x="16" y="${row.height - 17}" font-size="11" fill="${color}">${xml(statusLabel(row.status))}</text></g>`;
    })
    .join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" data-theme="${palette.name}" width="${layout.width}" height="${layout.height}" viewBox="0 0 ${layout.width} ${layout.height}" font-family="Inter Variable, DV Plex SC, sans-serif"><metadata>${xml(licenses)}</metadata><style>${fonts}</style><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M1 1 L9 5 L1 9" fill="none" stroke="#8b9aad"/></marker></defs><rect width="100%" height="100%" fill="${xml(palette.canvas)}"/>${lanes}${edges}${rows}</svg>`;
}
export function download(value: string, name: string, type: string): void {
  const url = URL.createObjectURL(new Blob([value], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
