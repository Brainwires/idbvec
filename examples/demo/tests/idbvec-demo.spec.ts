import { test, expect, type Page } from '@playwright/test'

const apps = [
  { name: 'react-app', url: 'http://localhost:5173' },
  { name: 'html-app', url: 'http://localhost:5174' },
]

// Helper to get stat card value by label text
function statValue(page: Page, label: string) {
  return page.locator('.stat-card', { hasText: label }).locator('.value')
}

// Helper to get the log/status area (React uses .status, HTML uses #log)
function logArea(page: Page) {
  return page.locator('.status, #log').first()
}

// Helper to get the metric select (first select on page)
function metricSelect(page: Page) {
  return page.locator('select').first()
}

// Helper to get the query select (last select on page)
function querySelect(page: Page) {
  return page.locator('select').last()
}

for (const app of apps) {
  test.describe(`${app.name}`, () => {
    test.beforeEach(async ({ page }) => {
      await page.goto(app.url)
      // Wait for WASM to load and buttons to enable
      await page.waitForFunction(() => {
        const btn = document.querySelector('button')
        return btn && !btn.disabled
      }, undefined, { timeout: 10_000 })
    })

    test('page loads with correct title and structure', async ({ page }) => {
      await expect(page.locator('h1')).toContainText('idbvec')
      await expect(page.locator('h1')).toContainText('Demo')

      // All 4 stat cards visible
      const cards = page.locator('.stat-card')
      await expect(cards).toHaveCount(4)
    })

    test('database initializes with 0 vectors and cosine metric', async ({ page }) => {
      await expect(statValue(page, 'Vectors')).toHaveText('0')
      await expect(statValue(page, 'Metric')).toHaveText('cosine')
      await expect(statValue(page, 'Dimensions')).toHaveText('4')
    })

    test('insert samples adds 10 vectors', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()

      await expect(statValue(page, 'Vectors')).toHaveText('10')
      await expect(logArea(page)).toContainText('Inserted 10 vectors')
      await expect(logArea(page)).toContainText('Total: 10')
    })

    test('search returns ranked results with correct structure', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      await page.getByRole('button', { name: 'Search (k=5)' }).click()

      // Results table with 5 rows
      const rows = page.locator('.results-table tbody tr')
      await expect(rows).toHaveCount(5)

      // First result is "cat" with distance 0
      await expect(rows.first().locator('td').nth(1)).toHaveText('cat')
      await expect(rows.first().locator('td').nth(2)).toHaveText('0.000000')

      // Second result is "dog"
      await expect(rows.nth(1).locator('td').nth(1)).toHaveText('dog')

      // Search time stat updated
      await expect(statValue(page, 'Search Time')).not.toHaveText('--')
    })

    test('search results are sorted by ascending distance', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      await page.getByRole('button', { name: 'Search (k=5)' }).click()

      const rows = page.locator('.results-table tbody tr')
      await expect(rows).toHaveCount(5)

      const distances: number[] = []
      for (let i = 0; i < 5; i++) {
        const text = await rows.nth(i).locator('td').nth(2).textContent()
        distances.push(parseFloat(text!))
      }

      for (let i = 1; i < distances.length; i++) {
        expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1])
      }
    })

    test('search with different query returns different first result', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      // Switch query to "car"
      await querySelect(page).selectOption('car')
      await page.getByRole('button', { name: 'Search (k=5)' }).click()

      const rows = page.locator('.results-table tbody tr')
      await expect(rows).toHaveCount(5)

      // First result is "car", second is "truck"
      await expect(rows.first().locator('td').nth(1)).toHaveText('car')
      await expect(rows.first().locator('td').nth(2)).toHaveText('0.000000')
      await expect(rows.nth(1).locator('td').nth(1)).toHaveText('truck')
    })

    test('get by ID retrieves vector and metadata', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      await page.getByRole('button', { name: 'Get by ID' }).click()

      const log = logArea(page)
      await expect(log).toContainText('get("cat")')
      await expect(log).toContainText('0.90')
      await expect(log).toContainText('0.80')
      await expect(log).toContainText('animal')
    })

    test('get by ID returns null for nonexistent vector', async ({ page }) => {
      await page.getByRole('button', { name: 'Get by ID' }).click()
      await expect(logArea(page)).toContainText('null')
    })

    test('delete removes vector and updates count', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      await page.getByRole('button', { name: 'Delete' }).click()

      await expect(statValue(page, 'Vectors')).toHaveText('9')
      await expect(logArea(page)).toContainText('delete("cat")')
      await expect(logArea(page)).toContainText('removed')
    })

    test('delete nonexistent vector reports not found', async ({ page }) => {
      await page.getByRole('button', { name: 'Delete' }).click()
      await expect(logArea(page)).toContainText('not found')
    })

    test('list IDs shows all stored vectors', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      await page.getByRole('button', { name: 'List IDs' }).click()

      const log = logArea(page)
      await expect(log).toContainText('Stored IDs (10)')
      await expect(log).toContainText('cat')
      await expect(log).toContainText('dog')
      await expect(log).toContainText('car')
    })

    test('clear removes all vectors', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      await page.getByRole('button', { name: 'Clear All' }).click()

      await expect(statValue(page, 'Vectors')).toHaveText('0')
      await expect(logArea(page)).toContainText('Database cleared')
    })

    test('standalone distance functions return correct values', async ({ page }) => {
      await page.getByRole('button', { name: 'Test Distance Fns' }).click()

      await expect(logArea(page)).toContainText('distance functions OK')

      // Check detailed output
      const pre = page.locator('pre').first()
      await expect(pre).toContainText('cosine_similarity: 0.000000')
      await expect(pre).toContainText('euclidean_distance: 1.414214')
      await expect(pre).toContainText('dot_product:        0.000000')
    })

    test('metric switching reinitializes database', async ({ page }) => {
      await metricSelect(page).selectOption('euclidean')

      // Wait for stat card to update (confirms reinitialization completed)
      await expect(statValue(page, 'Metric')).toHaveText('euclidean')
    })

    test('insert and search work after metric switch', async ({ page }) => {
      await metricSelect(page).selectOption('dotproduct')
      await expect(statValue(page, 'Metric')).toHaveText('dotproduct')

      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      await page.getByRole('button', { name: 'Search (k=5)' }).click()
      const rows = page.locator('.results-table tbody tr')
      await expect(rows).toHaveCount(5)
    })

    test('search on empty database is disabled or handles gracefully', async ({ page }) => {
      const searchBtn = page.getByRole('button', { name: 'Search (k=5)' })
      const isDisabled = await searchBtn.isDisabled()

      if (!isDisabled) {
        await searchBtn.click()
        // Should not crash
        await expect(logArea(page)).toBeVisible()
      }
    })

    test('full workflow: insert, search, delete, search again', async ({ page }) => {
      // Insert
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      // Search for cat
      await page.getByRole('button', { name: 'Search (k=5)' }).click()
      const rows = page.locator('.results-table tbody tr')
      await expect(rows).toHaveCount(5)
      await expect(rows.first().locator('td').nth(1)).toHaveText('cat')

      // Delete cat
      await page.getByRole('button', { name: 'Delete' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('9')

      // Search again — cat gone, dog should be first
      await page.getByRole('button', { name: 'Search (k=5)' }).click()
      await expect(rows).toHaveCount(5)
      await expect(rows.first().locator('td').nth(1)).toHaveText('dog')
    })

    test('upsert: reinserting keeps same vector count', async ({ page }) => {
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')

      // Insert again — should upsert, not duplicate
      await page.getByRole('button', { name: 'Insert Samples' }).click()
      await expect(statValue(page, 'Vectors')).toHaveText('10')
    })
  })
}
