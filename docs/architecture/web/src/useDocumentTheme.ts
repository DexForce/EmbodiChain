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

import { useEffect, useState } from 'react';
import { resolveDocsRoot } from './links';

type Theme = 'light' | 'dark';

function embeddingDocument(): Document | null {
  try {
    // Only the same-origin Sphinx embed owns our theme and full-screen link.
    if (
      resolveDocsRoot(new URL(location.href)) &&
      window.frameElement?.matches('.architecture-frame')
    ) {
      return window.frameElement.ownerDocument;
    }
  } catch {
    // Cross-origin hosts cannot expose their document; use standalone controls.
  }
  return null;
}

function initialTheme(host: Document | null): Theme {
  const inherited = host?.documentElement.dataset.theme;
  if (inherited === 'light' || inherited === 'dark') return inherited;
  const requested = new URLSearchParams(location.search).get('theme');
  if (requested === 'light' || requested === 'dark') return requested;
  try {
    return localStorage.getItem('architecture-academic-theme') === 'dark' ? 'dark' : 'light';
  } catch {
    return 'light';
  }
}

/** Follow the embedding Sphinx theme and preserve the reading state in outgoing links. */
export function useDocumentTheme(hash: string) {
  const [host] = useState(embeddingDocument);
  const [theme, setTheme] = useState<Theme>(() => initialTheme(host));
  useEffect(() => {
    if (!host) return;
    const sync = () => setTheme(initialTheme(host));
    const observer = new MutationObserver(sync);
    observer.observe(host.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    sync();
    return () => observer.disconnect();
  }, [host]);
  useEffect(() => {
    if (host) return;
    try {
      localStorage.setItem('architecture-academic-theme', theme);
    } catch {
      // Storage can be unavailable in private or restricted browser contexts.
    }
  }, [host, theme]);
  const url = new URL(location.href);
  url.hash = hash;
  url.searchParams.set('theme', theme);
  const shareUrl = url.href;
  useEffect(() => {
    const link = host?.querySelector<HTMLAnchorElement>('a.architecture-fullscreen');
    if (link) link.href = shareUrl;
  }, [host, shareUrl]);
  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    const updated = new URL(location.href);
    updated.searchParams.set('theme', next);
    history.replaceState(null, '', updated);
    setTheme(next);
  };
  return { theme, toggleTheme, embedded: host !== null, shareUrl };
}
