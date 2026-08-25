/**
 * The conversation window: transcript, the streaming draft + activity line,
 * the envelope's signal badge, and the composer (Enter sends, Shift+Enter
 * newline, Stop cancels).
 */
import { useEffect, useRef, useState, type KeyboardEvent } from 'react'
import type { Signal } from '../api/types'
import { Markdown } from './Markdown'
import type { ConversationState, Message } from './useConversation'

const SIGNAL_STYLE: Record<Signal, string> = {
  ok: 'bg-emerald-100 text-emerald-800',
  low_quality: 'bg-amber-100 text-amber-800',
  unjudged: 'bg-neutral-100 text-neutral-600',
  no_answer: 'bg-rose-100 text-rose-800',
}

/** Graded answers show the judge's score; the two unscored states say why. */
function signalLabel(signal: Signal, score: number | null): string {
  if (signal === 'unjudged') return 'unchecked'
  if (signal === 'no_answer') return 'no answer'
  return `confidence: ${score === null ? '?' : score.toFixed(2)}`
}

export function SignalBadge({ signal, score }: { signal: Signal; score: number | null }) {
  return (
    <span
      data-testid="signal"
      data-signal={signal}
      className={`rounded px-1.5 py-0.5 text-xs ${SIGNAL_STYLE[signal]}`}
    >
      {signalLabel(signal, score)}
    </span>
  )
}

function Bubble({ message }: { message: Message }) {
  const mine = message.role === 'user'
  return (
    <div className={`flex ${mine ? 'justify-end' : 'justify-start'}`}>
      <div
        data-role={message.role}
        className={`max-w-[80%] rounded-2xl px-4 py-2 text-sm ${
          mine
            ? 'whitespace-pre-wrap bg-blue-600 text-white'
            : 'bg-white text-neutral-900 shadow-sm'
        }`}
      >
        {mine ? (
          message.text
        ) : (
          <Markdown text={message.text} references={message.envelope?.references} />
        )}
        {message.envelope && (
          <div className="mt-2">
            <SignalBadge signal={message.envelope.signal} score={message.envelope.score} />
          </div>
        )}
      </div>
    </div>
  )
}

interface Props {
  state: ConversationState
  onSend: (text: string) => void
  onStop: () => void
}

export function ChatWindow({ state, onSend, onStop }: Props) {
  const [text, setText] = useState('')
  const bottom = useRef<HTMLDivElement>(null)
  const streaming = state.status === 'streaming'

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [state.messages.length, state.draft])

  const submit = () => {
    if (!text.trim()) return
    onSend(text)
    setText('')
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {state.messages.length === 0 && !streaming && (
          <div className="flex h-full items-center justify-center text-neutral-400">
            Ask a question to start a conversation.
          </div>
        )}
        {state.messages.map((m) => (
          <Bubble key={m.id} message={m} />
        ))}
        {streaming && (
          <div className="flex justify-start">
            <div
              data-testid="draft"
              className="max-w-[80%] rounded-2xl bg-white px-4 py-2 text-sm shadow-sm"
            >
              {state.draft ? (
                <Markdown text={state.draft} />
              ) : (
                <span className="text-neutral-400">{state.activity ?? 'thinking…'}</span>
              )}
              {state.draft && state.activity && (
                <div className="mt-1 text-xs text-neutral-400">{state.activity}</div>
              )}
            </div>
          </div>
        )}
        <div ref={bottom} />
      </div>
      {state.notice && (
        <div role="status" className="border-t border-neutral-200 bg-neutral-100 px-4 py-1 text-xs text-neutral-600">
          {state.notice}
        </div>
      )}
      <div className="flex gap-2 border-t border-neutral-200 bg-white p-3">
        <textarea
          aria-label="message"
          className="flex-1 resize-none rounded-lg border border-neutral-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={2}
          placeholder={streaming ? 'Add to the current answer…' : 'Ask a question…'}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
        />
        {streaming ? (
          <button
            type="button"
            onClick={onStop}
            className="rounded-lg bg-neutral-800 px-4 text-sm text-white hover:bg-neutral-700"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={submit}
            disabled={!text.trim()}
            className="rounded-lg bg-blue-600 px-4 text-sm text-white hover:bg-blue-500 disabled:opacity-40"
          >
            Send
          </button>
        )}
      </div>
    </div>
  )
}
