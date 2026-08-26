import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { setupServer } from 'msw/node'
import App from '../App'
import { navigation } from '../api/auth'
import { TOKEN_KEY } from '../api/client'
import { http, HttpResponse } from '../test/sse'

const b64url = (o: object) =>
  btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
const TOKEN = `h.${b64url({ sub: 'alice', role: 'user', exp: 1800000000 })}.s`

const health = (auth: boolean, sso: boolean) =>
  http.get('/health', () => HttpResponse.json({ status: 'ok', spec: 's', auth, sso }))

const server = setupServer(
  http.get('/v1/conversations', () => HttpResponse.json({ conversations: [] })),
)
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => {
  server.resetHandlers()
  localStorage.clear()
  window.history.replaceState(null, '', '/')
  vi.restoreAllMocks()
})
afterAll(() => server.close())

test('dev mode: no login UI, just "anonymous"', async () => {
  server.use(health(false, false))
  render(<App />)
  expect(await screen.findByText('anonymous')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Log in' })).toBeNull()
})

test('auth required, nobody signed in: the login screen, one action -> /sso', async () => {
  server.use(health(true, true))
  const go = vi.spyOn(navigation, 'go').mockImplementation(() => {})
  render(<App />)
  const user = userEvent.setup()
  const button = await screen.findByRole('button', { name: 'Log in as Employee' })
  expect(screen.queryByLabelText('message')).toBeNull() // no chat UI behind it
  await user.click(button)
  expect(go).toHaveBeenCalledWith('/sso?next=%2F')
})

test('SSO available but auth off (dev with saml): chat renders, "Log in" offered in the panel', async () => {
  server.use(health(false, true))
  render(<App />)
  expect(await screen.findByRole('button', { name: 'Log in' })).toBeInTheDocument()
  expect(screen.getByLabelText('message')).toBeInTheDocument()
})

test('back from the ACS: token captured, principal shown, logout drops it', async () => {
  server.use(health(true, true))
  window.history.replaceState(null, '', `/#token=${TOKEN}`)
  render(<App />)
  expect(await screen.findByText('alice')).toBeInTheDocument()
  expect(localStorage.getItem(TOKEN_KEY)).toBe(TOKEN)
  expect(window.location.hash).toBe('')
  const user = userEvent.setup()
  await user.click(screen.getByRole('button', { name: 'Log out' }))
  expect(localStorage.getItem(TOKEN_KEY)).toBeNull()
  // auth is required here, so logging out lands on the login screen
  expect(await screen.findByRole('button', { name: 'Log in as Employee' })).toBeInTheDocument()
})
