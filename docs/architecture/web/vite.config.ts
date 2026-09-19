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

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
const snapshotPath =
  process.env.ARCHITECTURE_DATA_PATH ??
  fileURLToPath(new URL('../generated/architecture.json', import.meta.url));
export default defineConfig({
  base: './',
  plugins: [
    react(),
    {
      name: 'architecture-snapshot',
      configureServer(server) {
        server.middlewares.use((request, response, next) => {
          if (request.url?.split('?')[0] !== '/architecture.json') return next();
          try {
            response.setHeader('Content-Type', 'application/json');
            response.end(readFileSync(snapshotPath));
          } catch {
            response.statusCode = 500;
            response.end('Snapshot unavailable');
          }
        });
      },
      generateBundle() {
        this.emitFile({
          type: 'asset',
          fileName: 'architecture.json',
          source: readFileSync(snapshotPath, 'utf8'),
        });
      },
    },
  ],
  test: { include: ['tests/*.test.ts'] },
});
