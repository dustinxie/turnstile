/**
 * Left column: the principal's conversations, newest first, plus "New chat".
 * Pure presentation — the list and the active id come from App.
 */
import type { ConversationSummary } from '../api/types'

interface Props {
  items: ConversationSummary[]
  activeId: string
  error: string | null
  onSelect: (id: string) => void
  onNew: () => void
  /** login state: who is signed in (null = nobody / dev mode) */
  principal: { sub: string } | null
  canLogin: boolean // the deployment has an SSO route and nobody is signed in
  onLogin: () => void
  onLogout: () => void
}

export function HistoryPanel({
  items,
  activeId,
  error,
  onSelect,
  onNew,
  principal,
  canLogin,
  onLogin,
  onLogout,
}: Props) {
  return (
    <aside
      aria-label="conversation history"
      className="flex w-64 shrink-0 flex-col border-r border-neutral-200 bg-white"
    >
      <div className="flex items-center justify-between border-b border-neutral-200 px-4 py-3">
        <span className="text-sm font-semibold">turnstile</span>
        <button
          type="button"
          onClick={onNew}
          className="rounded-md bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-500"
        >
          New chat
        </button>
      </div>
      <nav aria-label="conversations" className="flex-1 overflow-y-auto p-2">
        {error && <div className="px-2 py-1 text-xs text-rose-700">{error}</div>}
        {!error && items.length === 0 && (
          <div className="px-2 py-1 text-sm text-neutral-500">No conversations yet.</div>
        )}
        <ul className="space-y-0.5">
          {items.map((c) => {
            const active = c.conversation_id === activeId
            return (
              <li key={c.conversation_id}>
                <button
                  type="button"
                  aria-current={active ? 'true' : undefined}
                  onClick={() => onSelect(c.conversation_id)}
                  className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-sm hover:bg-neutral-100 ${
                    active ? 'bg-neutral-100 font-medium' : ''
                  }`}
                >
                  {c.in_flight && (
                    <span
                      aria-label="answering"
                      title="a turn is still running"
                      className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-blue-500"
                    />
                  )}
                  <span className="truncate">{c.title || 'Untitled'}</span>
                </button>
              </li>
            )
          })}
        </ul>
      </nav>
      <div
        aria-label="account"
        className="flex items-center justify-between border-t border-neutral-200 px-4 py-2 text-xs"
      >
        {principal ? (
          <>
            <span className="truncate text-neutral-700" title={principal.sub}>
              {principal.sub}
            </span>
            <button type="button" onClick={onLogout} className="text-neutral-500 hover:underline">
              Log out
            </button>
          </>
        ) : canLogin ? (
          <button type="button" onClick={onLogin} className="text-blue-600 hover:underline">
            Log in
          </button>
        ) : (
          <span className="text-neutral-400">anonymous</span>
        )}
      </div>
    </aside>
  )
}
