/**
 * Login state for the header: who is logged in (from the stored token),
 * whether the deployment even has a login route (from /health), and logout.
 * Captures a "#token=" fragment on first render — the ACS return path.
 */
import { useCallback, useEffect, useState } from 'react'
import {
  captureTokenFromFragment,
  loginUrl,
  logout,
  navigation,
  principal,
  ssoAvailable,
} from '../api/auth'
import { getHealth } from '../api/client'

export function useAuth() {
  const [who, setWho] = useState(() => {
    captureTokenFromFragment()
    return principal()
  })
  // null until /health answers: the app renders nothing auth-related meanwhile
  const [health, setHealth] = useState<{ auth: boolean; sso: boolean } | null>(null)

  useEffect(() => {
    getHealth()
      .then((h) => {
        ssoAvailable.value = h.sso
        setHealth({ auth: h.auth, sso: h.sso })
      })
      .catch(() => setHealth({ auth: false, sso: false }))
  }, [])

  const doLogout = useCallback(() => {
    logout()
    setWho(null)
  }, [])
  const doLogin = useCallback(() => navigation.go(loginUrl()), [])

  const sso = health?.sso ?? false
  return {
    principal: who,
    canLogin: sso && who === null,
    // the deployment requires a token and we have none: show the login screen
    mustLogin: health !== null && health.auth && who === null,
    // nothing authenticated may be fetched before /health has answered
    ready: health !== null,
    login: doLogin,
    logout: doLogout,
  }
}
