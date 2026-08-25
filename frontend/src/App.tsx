/**
 * The one screen: a history panel on the left, the conversation on the right.
 * F1 ships the empty shell; F2 fills the chat window, F4 the history panel.
 */
export default function App() {
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
        <div className="flex flex-1 items-center justify-center text-neutral-400">
          Ask a question to start a conversation.
        </div>
      </main>
    </div>
  )
}
