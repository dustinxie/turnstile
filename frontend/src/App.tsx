/**
 * The one screen: the history panel on the left, the conversation on the
 * right. The client picks conversation ids (uuid); the first POST claims one,
 * and the panel refreshes when a turn settles so the new chat appears titled.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { ChatWindow } from './chat/ChatWindow'
import { useConversation } from './chat/useConversation'
import { HistoryPanel } from './history/HistoryPanel'
import { useConversationList } from './history/useConversationList'

export default function App() {
  const [initialId] = useState(() => crypto.randomUUID())
  const { state, send, stop, load, start } = useConversation(initialId)
  const { items, error, refresh } = useConversationList()

  // a turn just settled -> the list may have a new entry / new title
  const wasStreaming = useRef(false)
  useEffect(() => {
    if (wasStreaming.current && state.status === 'idle') void refresh()
    wasStreaming.current = state.status === 'streaming'
  }, [state.status, refresh])

  const onSelect = useCallback(
    (id: string) => {
      if (id !== state.conversationId) void load(id)
    },
    [load, state.conversationId],
  )
  const onNew = useCallback(() => start(crypto.randomUUID()), [start])

  return (
    <div className="flex h-full bg-neutral-50 text-neutral-900">
      <HistoryPanel
        items={items}
        activeId={state.conversationId}
        error={error}
        onSelect={onSelect}
        onNew={onNew}
      />
      <main aria-label="conversation" className="flex min-w-0 flex-1 flex-col">
        <ChatWindow state={state} onSend={send} onStop={stop} />
      </main>
    </div>
  )
}
