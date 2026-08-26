/**
 * Login state on the client. The service issues a JWT (SSO -> ACS -> 302 to
 * us with "#token=<jwt>"); we keep it in localStorage and attach it as a
 * Bearer header on every call (client.ts). The browser never does this by
 * itself — that's the cookie model we deliberately did not choose.
 */
import { TOKEN_KEY, getToken } from './client'

export interface Principal {
  sub: string
  role: string
  exp: number // unix seconds
}

/** On page load: a "#token=..." fragment means we just came back from the
 * ACS. Store it and strip it from the address bar (history, bookmarks). */
export function captureTokenFromFragment(): boolean {
  const match = /(?:^#|&)token=([^&]+)/.exec(window.location.hash)
  if (!match) return false
  try {
    localStorage.setItem(TOKEN_KEY, decodeURIComponent(match[1]))
  } catch {
    return false
  }
  window.history.replaceState(null, '', window.location.pathname + window.location.search)
  return true
}

/** Decode the stored token's payload for display (NOT for trust — the
 * service verifies the signature; this is just base64). */
export function principal(): Principal | null {
  const token = getToken()
  if (!token) return null
  try {
    const payload = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')
    const claims = JSON.parse(atob(payload)) as Partial<Principal>
    if (typeof claims.sub !== 'string') return null
    return { sub: claims.sub, role: claims.role ?? 'user', exp: claims.exp ?? 0 }
  } catch {
    return null
  }
}

export function logout(): void {
  try {
    localStorage.removeItem(TOKEN_KEY)
  } catch {
    /* storage unavailable: nothing to drop */
  }
}

/** Where to send the browser to log in; `next` = the path to come back to
 * (rides SAML RelayState; the service accepts same-origin paths only). */
export function loginUrl(next: string = window.location.pathname): string {
  return `/sso?next=${encodeURIComponent(next)}`
}

/** Indirection so tests can observe navigation (jsdom cannot navigate). */
export const navigation = {
  go(url: string): void {
    window.location.assign(url)
  },
}

/** Set from /health by useAuth: is there an SSO route to send people to? */
export const ssoAvailable = { value: false }

/** A 401 means the token is missing/expired: drop it and go log in — via
 * SSO when the deployment has it, else back to the app root, where the
 * login screen (or dev-mode chat) takes over. */
export function onUnauthorized(): void {
  logout()
  navigation.go(ssoAvailable.value ? loginUrl() : '/')
}
