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

test('navigate both views, inspect evidence, and restore a shared selection', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '架构浏览器', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Task Program', exact: true }).click();
  await page.getByRole('button', { name: '选择 AtomicActionEngine', exact: true }).click();
  const details = page.getByRole('complementary', { name: '节点详情' });
  await expect(
    details.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
  ).toBeVisible();
  await expect(details.getByRole('link', { name: '查看源码', exact: true })).toHaveAttribute(
    'href',
    /blob\/3224ac1e/,
  );
  await details.getByRole('button', { name: '持有 MotionGenerator', exact: true }).click();
  await expect(
    details.getByRole('button', { name: '被 AtomicActionEngine 持有', exact: true }),
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
  await page.getByRole('searchbox', { name: '搜索模块' }).fill('no-match-xyz');
  await expect(page.getByText('没有找到匹配的模块')).toBeVisible();
  await page.getByRole('searchbox', { name: '搜索模块' }).fill('');
  await page.getByRole('button', { name: '关系类型', exact: true }).click();
  await page.getByRole('button', { name: '清空关系筛选', exact: true }).click();
  await expect(
    page.getByRole('heading', { name: 'AtomicActionEngine', exact: true }),
  ).toBeVisible();
  await expect(page.locator('.react-flow__edge')).toHaveCount(0);
});

test('narrow screens keep navigation and details accessible', async ({ page }) => {
  await page.setViewportSize({ width: 700, height: 900 });
  await page.goto('/');
  await page.getByRole('button', { name: '显示模块列表', exact: true }).click();
  await page.getByRole('button', { name: '选择 Robot', exact: true }).click();
  await expect(page.getByRole('complementary', { name: '节点详情' })).toBeVisible();
  await page.getByRole('button', { name: '关闭详情', exact: true }).click();
  await expect(page.getByRole('button', { name: '适应画布', exact: true })).toBeVisible();
});
