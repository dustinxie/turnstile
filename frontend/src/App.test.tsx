import { render, screen } from '@testing-library/react'
import { setupServer } from 'msw/node'
import App from './App'
import { http, HttpResponse } from './test/sse'

const server = setupServer(
  http.get('/v1/conversations', () => HttpResponse.json({ conversations: [] })),
)
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterAll(() => server.close())

test('renders the two-column shell', async () => {
  render(<App />)
  expect(screen.getByRole('complementary', { name: /conversation history/i })).toBeInTheDocument()
  expect(screen.getByRole('main', { name: /conversation/i })).toBeInTheDocument()
  expect(await screen.findByText('No conversations yet.')).toBeInTheDocument()
})
