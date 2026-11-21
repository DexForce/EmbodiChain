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

import { useReactFlow, useViewport } from "@xyflow/react";
import { Minus, Plus, Maximize } from "lucide-react";

export interface GraphControlLabels {
  zoomOut: string;
  actualSize: string;
  zoomIn: string;
  fit: string;
}

const english: GraphControlLabels = {
  zoomOut: "Zoom out",
  actualSize: "Read at 100 percent",
  zoomIn: "Zoom in",
  fit: "Fit to view",
};

/** Data-independent controls used by documentation and recorded-task diagrams. */
export function GraphViewportControls({
  labels = english,
}: {
  labels?: GraphControlLabels;
}) {
  const flow = useReactFlow();
  const { zoom } = useViewport();
  return (
    <div className="canvas-tools graph-ui-controls">
      <button
        aria-label={labels.zoomOut}
        title={labels.zoomOut}
        onClick={() => flow.zoomOut({ duration: 180 })}
      >
        <Minus size={16} />
      </button>
      <button
        className="reading-size"
        aria-label={labels.actualSize}
        title={labels.actualSize}
        onClick={() => flow.zoomTo(1, { duration: 180 })}
      >
        {Math.round(zoom * 100)}%
      </button>
      <button
        aria-label={labels.zoomIn}
        title={labels.zoomIn}
        onClick={() => flow.zoomIn({ duration: 180 })}
      >
        <Plus size={16} />
      </button>
      <i />
      <button
        aria-label={labels.fit}
        title={labels.fit}
        onClick={() => flow.fitView({ padding: 0.07, duration: 350 })}
      >
        <Maximize size={15} />
      </button>
    </div>
  );
}
