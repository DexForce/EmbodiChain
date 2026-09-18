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

import { describe, expect, it } from 'vitest';
import { documentationUrl, resolveDocsRoot } from '../src/links';

describe('version-local documentation links', () => {
  for (const prefix of ['/', '/main/', '/v0.2.4/', '/EmbodiChain/main/']) {
    it(`keeps links within ${prefix}`, () => {
      const page = new URL(
        `https://example.org${prefix}_static/architecture/index.html?docsRoot=../../`,
      );
      const root = resolveDocsRoot(page)!;
      expect(documentationUrl('api_reference/engine', root)).toBe(
        `https://example.org${prefix}api_reference/engine.html`,
      );
    });
  }
  it('retains standalone source links without an explicit docs root', () => {
    expect(resolveDocsRoot(new URL('https://example.org/index.html'))).toBeNull();
    expect(
      resolveDocsRoot(new URL('https://example.org/index.html?docsRoot=https://other.example/')),
    ).toBeNull();
  });
});
