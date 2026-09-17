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

import { test, expect } from '@playwright/test';

test('embedded explorer follows document theme and opens the current view full-screen', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.addInitScript(() => localStorage.setItem('mode', 'dark'));
  await page.goto('/main/overview/architecture/index.html');
  const frame = page.frameLocator('iframe.architecture-frame');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(frame.locator('.app')).toHaveAttribute('data-theme', 'dark');
  const menu = frame.getByRole('button', { name: 'Show module list', exact: true });
  if (await menu.isVisible()) await menu.click();
  await frame.getByRole('button', { name: 'Select AtomicActionEngine', exact: true }).click();
  await frame.getByRole('button', { name: 'Direct neighbours only', exact: true }).click();
  // Exercise the theme's DOM contract, including changes after iframe load.
  await page.locator('html').evaluate((element) => element.setAttribute('data-theme', 'light'));
  await expect(frame.locator('.app')).toHaveAttribute('data-theme', 'light');
  await frame.getByRole('button', { name: 'Share view', exact: true }).click();
  await expect(frame.getByRole('button', { name: 'Copied', exact: true })).toBeVisible();
  const shared = await page.evaluate(() => navigator.clipboard.readText());
  const fullScreen = page.getByRole('link', { name: 'Open full-screen explorer' });
  await expect(fullScreen).toHaveAttribute('href', shared);
  const url = new URL(shared);
  expect(url.pathname).toBe('/main/_static/architecture/index.html');
  expect(url.searchParams.get('theme')).toBe('light');
  expect(new URLSearchParams(url.hash.slice(1)).get('node')).toBe('atomic-engine');
  expect(new URLSearchParams(url.hash.slice(1)).get('focus')).toBe('1');
  await fullScreen.click();
  await expect(page.locator('.app')).toHaveAttribute('data-theme', 'light');
  await expect(
    page.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await page.reload();
  await expect(page.locator('.app')).toHaveAttribute('data-theme', 'dark');
});

test('architecture entry uses the reading width without horizontal page overflow', async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto('/overview/architecture/index.html');
  const frame = page.locator('iframe.architecture-frame');
  const desktop = await frame.boundingBox();
  expect(desktop!.width).toBeGreaterThan(820);
  await page.setViewportSize({ width: 390, height: 844 });
  const mobile = await frame.boundingBox();
  expect(mobile!.width).toBeLessThanOrEqual(390);
  expect(mobile!.height).toBeGreaterThanOrEqual(560);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  const embedded = page.frameLocator('iframe.architecture-frame');
  await embedded.getByRole('button', { name: 'Show module list', exact: true }).click();
  await embedded.getByRole('button', { name: 'Select Robot', exact: true }).click();
  await expect(embedded.getByRole('heading', { name: 'Robot', exact: true })).toBeVisible();
});

for (const prefix of ['/', '/main/', '/v0.2.4/', '/EmbodiChain/main/']) {
  test(`embedded and full-screen links stay within ${prefix}`, async ({ page, request }) => {
    await page.goto(`${prefix}overview/architecture/index.html`);
    await expect(
      page.getByRole('heading', { name: 'Architecture Explorer', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('heading', { name: 'Module reference', exact: true }),
    ).toBeVisible();
    const fullScreen = await page
      .getByRole('link', { name: 'Open full-screen explorer' })
      .getAttribute('href');
    const fullUrl = new URL(fullScreen!, page.url());
    const response = await request.get(`${prefix}_static/architecture/architecture.json`);
    expect(response.ok()).toBe(true);
    const snapshot = await response.json();
    const frame = page.frameLocator('iframe.architecture-frame');
    const menu = frame.getByRole('button', { name: 'Show module list', exact: true });
    if (await menu.isVisible()) await menu.click();
    await frame.getByRole('button', { name: 'Select AtomicActionEngine', exact: true }).click();
    await expect(
      frame.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
    ).toBeVisible();
    await expect(frame.getByRole('link', { name: 'View source', exact: true })).toHaveAttribute(
      'href',
      new RegExp(`/blob/${snapshot.revision}/`),
    );
    const documentation = frame.getByRole('link', {
      name: 'AtomicActionEngine documentation',
      exact: true,
    });
    await expect(documentation).toHaveAttribute('target', '_parent');
    await expect(documentation).toHaveAttribute(
      'href',
      new URL(
        `${prefix}api_reference/embodichain/embodichain.lab.sim.atomic_actions.html`,
        page.url(),
      ).href,
    );
    await documentation.click();
    await expect(page).toHaveURL(
      new RegExp(`${prefix}api_reference/embodichain/embodichain.lab.sim.atomic_actions.html$`),
    );
    fullUrl.hash = 'view=task-program&node=atomic-engine&focus=1';
    await page.goto(fullUrl.href);
    await expect(
      page.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole('button', { name: 'Direct neighbours only', exact: true }),
    ).toHaveAttribute('aria-pressed', 'true');
    await page.reload();
    await expect(
      page.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
    ).toBeVisible();
  });
}

test('unavailable snapshot shows an actionable error instead of an empty canvas', async ({
  page,
}) => {
  await page.route('**/architecture.json', (route) =>
    route.fulfill({ status: 503, body: 'Unavailable' }),
  );
  await page.goto('/_static/architecture/index.html');
  await expect(page.getByRole('alert')).toContainText('Snapshot request failed (503)');
});

test('invalid snapshot shows its format error', async ({ page }) => {
  await page.route('**/architecture.json', (route) =>
    route.fulfill({ contentType: 'application/json', body: '{"schema_version":99}' }),
  );
  await page.goto('/_static/architecture/index.html');
  await expect(page.getByRole('alert')).toContainText('unsupported or invalid format');
});
