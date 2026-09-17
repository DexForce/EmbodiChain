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
import snapshot from '../../generated/architecture.json' with { type: 'json' };

test('navigate both views, inspect evidence, and restore a shared selection', async ({ page }) => {
  await page.goto('/');
  await expect(
    page.getByRole('heading', { name: 'Architecture Explorer', exact: true }),
  ).toBeVisible();
  await page.getByRole('button', { name: 'Task Program', exact: true }).click();
  await page.getByRole('button', { name: 'Select AtomicActionEngine', exact: true }).click();
  const details = page.getByRole('complementary', { name: 'Node details' });
  await expect(
    details.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
  ).toBeVisible();
  await expect(details.getByRole('link', { name: 'View source', exact: true })).toHaveAttribute(
    'href',
    new RegExp(`/blob/${snapshot.revision}/`),
  );
  await details.getByRole('button', { name: 'Holds MotionGenerator', exact: true }).click();
  await expect(
    details.getByRole('button', {
      name: 'Incoming: AtomicActionEngine holds this module',
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    details.getByRole('heading', { name: 'MotionGenerator', exact: true }),
  ).toBeVisible();
  await page.reload();
  await expect(
    details.getByRole('heading', { name: 'MotionGenerator', exact: true }),
  ).toBeVisible();
  await page.goBack();
  await expect(
    details.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
  ).toBeVisible();
});

test('search has clear empty state and relationship filters keep selected details', async ({
  page,
}) => {
  await page.goto('/#view=task-program&node=atomic-engine');
  await page.getByRole('searchbox', { name: 'Search modules' }).fill('no-match-xyz');
  await expect(page.getByText('No matching modules')).toBeVisible();
  await page.getByRole('searchbox', { name: 'Search modules' }).fill('');
  await page.getByRole('button', { name: 'Relationships', exact: true }).click();
  await page.getByRole('button', { name: 'Clear relationship filters', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
  ).toBeVisible();
  await expect(page.locator('.react-flow__edge')).toHaveCount(0);
});

test('narrow screens keep navigation and details accessible', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Show module list', exact: true }).click();
  await page.getByRole('button', { name: 'Select Robot', exact: true }).click();
  await expect(page.getByRole('complementary', { name: 'Node details' })).toBeVisible();
  await page.getByRole('button', { name: 'Close details', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Fit to view', exact: true })).toBeVisible();
});

test('reading view keeps text legible and local focus survives navigation', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/#view=task-program&node=atomic-engine');
  await expect(page.locator('.canvas-tools')).toContainText('100%');
  await page.getByRole('button', { name: 'Direct neighbours only', exact: true }).click();
  await expect(page.locator('.react-flow__node-module')).toHaveCount(5);
  await expect(page.locator('.react-flow__edge')).toHaveCount(7);
  await expect(page.locator('.react-flow__node-module[data-id="program-compiler"]')).toHaveCount(0);
  await page.reload();
  await expect(
    page.getByRole('button', { name: 'Direct neighbours only', exact: true }),
  ).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'Relationships', exact: true }).click();
  await page.getByRole('button', { name: 'Clear relationship filters', exact: true }).click();
  await expect(page.locator('.react-flow__node-module')).toHaveCount(1);
  await expect(page.getByText('No direct relationships match these filters.')).toBeVisible();
  await page.getByRole('button', { name: 'System overview', exact: true }).click();
  await expect(
    page.getByRole('button', { name: 'Direct neighbours only', exact: true }),
  ).toBeDisabled();
  await page.getByRole('button', { name: 'Select Devices', exact: true }).click();
  await expect(page.getByText('Relationships not yet mapped in this view.')).toBeVisible();
});

test('mobile readers can hide details, focus neighbours, and reopen the same selection', async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/#view=task-program&node=atomic-engine');
  await page.getByRole('button', { name: 'Close details', exact: true }).click();
  const focus = page.getByRole('button', { name: 'Direct neighbours only', exact: true });
  await expect(focus).toBeEnabled();
  await focus.click();
  await expect(page.locator('.react-flow__node-module')).toHaveCount(5);
  await page.getByRole('button', { name: 'Open node details', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
  ).toBeVisible();
});

test('large neighbourhoods keep the selected module readable above the fold', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto('/#view=overview&node=embodied-env');
  const title = page.locator('.react-flow__node-module[data-id="embodied-env"] .module-name');
  const titleIsInsideCanvas = async () => {
    const box = await title.boundingBox();
    const canvas = await page.locator('.canvas-shell').boundingBox();
    return !!box && !!canvas && box.y >= canvas.y && box.y + box.height <= canvas.y + canvas.height;
  };
  await expect.poll(titleIsInsideCanvas).toBe(true);
  await page.getByRole('button', { name: 'Direct neighbours only', exact: true }).click();
  await expect(page.locator('.react-flow__node-module')).toHaveCount(9);
  await expect.poll(titleIsInsideCanvas).toBe(true);
});

test('module cards expose a keyboard-accessible reading panel with documentation first', async ({
  page,
}) => {
  await page.goto('/');
  const card = page.getByRole('button', { name: 'Explore CLI', exact: true });
  await card.focus();
  await page.keyboard.press('Enter');
  const details = page.getByRole('complementary', { name: 'Node details' });
  await expect(details.getByRole('heading', { name: 'CLI', exact: true })).toBeVisible();
  await expect(details.getByRole('heading', { name: 'About this module' })).toBeVisible();
  await expect(details.getByRole('link', { name: 'CLI reference', exact: true })).toBeVisible();
  const documentation = await details.locator('.documentation-links').boundingBox();
  const relationships = await details.locator('.relations-section').boundingBox();
  expect(documentation!.y).toBeLessThan(relationships!.y);
  await page.getByRole('button', { name: 'Select Official tasks', exact: true }).click();
  await expect(details.getByRole('link', { name: 'Supported tasks', exact: true })).toHaveAttribute(
    'href',
    /resources\/task\/index\.rst$/,
  );
});
