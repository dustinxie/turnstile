/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The frontend is a static SPA that talks to the turnstile service. In dev,
// Vite proxies the API paths to a locally running uvicorn so the browser sees
// one origin (no CORS); in prod the container / nginx serves both.
// Override the backend with VITE_API_TARGET=http://127.0.0.1:8001 npm run dev
// when :8000 is taken (e.g. a running container).
const target = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { '/v1': target, '/health': target, '/sso': target },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
})
