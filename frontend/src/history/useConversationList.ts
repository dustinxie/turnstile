/**
 * The history panel's data: the principal's conversations from the service.
 * Refreshed on demand (mount, after a turn completes, after switching) — no
 * polling; a turn running detached shows via `in_flight` on the next refresh.
 */
import { useCallback, useEffect, useState } from 'react'
import { listConversations } from '../api/client'
import type { ConversationSummary } from '../api/types'

export function useConversationList() {
  const [items, setItems] = useState<ConversationSummary[]>([])
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    try {
      setItems(await listConversations())
      setError(null)
    } catch (e) {
      setError((e as Error).message || 'could not load conversations')
    }
  }, [])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect -- setState runs after the fetch resolves, not synchronously
    void refresh()
  }, [refresh])

  return { items, error, refresh }
}
