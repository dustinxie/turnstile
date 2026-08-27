import {
  captureTokenFromFragment,
  loginUrl,
  navigation,
  onUnauthorized,
  principal,
  ssoAvailable,
} from './auth'
import { TOKEN_KEY } from './client'

// a JWT-shaped token: header.payload.signature (payload is plain base64url)
const b64url = (o: object) =>
  btoa(JSON.stringify(o)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
const TOKEN = `eyJhbGciOiJIUzI1NiJ9.${b64url({ sub: 'alice', role: 'admin', exp: 1800000000 })}.sig`

afterEach(() => {
  localStorage.clear()
  window.history.replaceState(null, '', '/')
  vi.restoreAllMocks()
})

test('captures #token= from the ACS redirect and strips it from the URL', () => {
  window.history.replaceState(null, '', `/chat?x=1#token=${TOKEN}`)
  expect(captureTokenFromFragment()).toBe(true)
  expect(localStorage.getItem(TOKEN_KEY)).toBe(TOKEN)
  expect(window.location.hash).toBe('') // never lingers in history/bookmarks
  expect(window.location.pathname + window.location.search).toBe('/chat?x=1')
  expect(captureTokenFromFragment()).toBe(false) // nothing on a plain load
})

test('principal is decoded for display from the stored token', () => {
  expect(principal()).toBeNull()
  localStorage.setItem(TOKEN_KEY, TOKEN)
  expect(principal()).toEqual({ sub: 'alice', role: 'admin', exp: 1800000000 })
  localStorage.setItem(TOKEN_KEY, 'garbage')
  expect(principal()).toBeNull()
})

test('a 401 drops the token and sends the browser to login with the return path', () => {
  const go = vi.spyOn(navigation, 'go').mockImplementation(() => {})
  localStorage.setItem(TOKEN_KEY, TOKEN)
  window.history.replaceState(null, '', '/chat/c1')
  ssoAvailable.value = true
  onUnauthorized()
  expect(localStorage.getItem(TOKEN_KEY)).toBeNull()
  expect(go).toHaveBeenCalledWith('/v1/sso?next=%2Fchat%2Fc1')
  expect(loginUrl('/x')).toBe('/v1/sso?next=%2Fx')
  // no SSO route on this deployment: back to the root, the login screen takes over
  ssoAvailable.value = false
  onUnauthorized()
  expect(go).toHaveBeenLastCalledWith('/')
})
