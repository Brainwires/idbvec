import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './tests',
  timeout: 30_000,
  retries: 0,
  use: {
    headless: true,
    browserName: 'chromium',
  },
  webServer: [
    {
      command: 'npm run dev',
      cwd: './react-app',
      port: 5173,
      reuseExistingServer: true,
    },
    {
      command: 'npm run dev',
      cwd: './html-app',
      port: 5174,
      reuseExistingServer: true,
    },
  ],
})
