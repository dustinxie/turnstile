/**
 * The one screen: a history panel on the left, the conversation on the right.
 * F2: the chat window is live against a fresh conversation id per page load
 * (the client picks the id; the first POST claims it). F4 fills the panel.
 */
import { useState } from 'react'
import { ChatWindow } from './chat/ChatWindow'
import { useConversation } from './chat/useConversation'

export default function App() {
  const [conversationId] = useState(() => crypto.randomUUID())
  const { state, send, stop } = useConversation(conversationId)

  return (
    <div className="flex h-full bg-neutral-50 text-neutral-900">
      <aside
        aria-label="conversation history"
        className="flex w-64 shrink-0 flex-col border-r border-neutral-200 bg-white"
      >
        <div className="border-b border-neutral-200 px-4 py-3 text-sm font-semibold">
          turnstile
        </div>
        <div className="flex-1 overflow-y-auto p-2 text-sm text-neutral-500">
          No conversations yet.
        </div>
      </aside>
      <main aria-label="conversation" className="flex min-w-0 flex-1 flex-col">
        <ChatWindow state={state} onSend={send} onStop={stop} />
      </main>
    </div>
  )
}
