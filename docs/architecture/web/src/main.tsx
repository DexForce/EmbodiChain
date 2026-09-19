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

import { StrictMode, useEffect, useState } from 'react';
import Ajv2020 from 'ajv/dist/2020';
import schema from '../../architecture.schema.json';
import type { ArchitectureSnapshot } from './types';
import { createRoot } from 'react-dom/client';
import '@fontsource-variable/inter/wght.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import '@xyflow/react/dist/style.css';
import './styles.css';
import App from './App';
const validate = new Ajv2020().compile<ArchitectureSnapshot>(schema);
function LoadExplorer() {
  const [data, setData] = useState<ArchitectureSnapshot | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    fetch(new URL('architecture.json', location.href), { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Snapshot request failed (${response.status}).`);
        const snapshot: unknown = await response.json();
        if (!validate(snapshot))
          throw new Error('The architecture snapshot has an unsupported or invalid format.');
        setData(snapshot);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted)
          setError(
            cause instanceof Error ? cause.message : 'Unable to load the architecture snapshot.',
          );
      });
    return () => controller.abort();
  }, []);
  if (error)
    return (
      <main className="startup-state" role="alert">
        <h1>Architecture Explorer unavailable</h1>
        <p>{error}</p>
        <p>Reload the page or return to the documentation overview.</p>
      </main>
    );
  if (!data)
    return (
      <main className="startup-state" role="status">
        Loading architecture snapshot…
      </main>
    );
  return <App data={data} />;
}
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LoadExplorer />
  </StrictMode>,
);
