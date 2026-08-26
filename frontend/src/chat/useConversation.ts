/**
 * One conversation's UI state, driven by the turn stream. The reducer is the
 * SSE tagged union's mirror: one case per event the UI reacts to.
 *
 * Rendering is ENVELOPE-DRIVEN: text_delta paints a provisional bubble as
 * tokens arrive (only when the deployment streams text — a judged deployment
 * withholds it), and the envelope's `answer` REPLACES that bubble at the end.
 * Both deployment modes therefore render identically, and the final bubble
 * carries the References section the service appended.
 */
import { useCallback, useReducer, useRef } from 'react'
import { cancelTurn, getConversation, postMessage } from '../api/client'
import type { Envelope, TurnEvent } from '../api/types'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text: string
  envelope?: Envelope // set on the final assistant message of a turn
}

export interface ConversationState {
  conversationId: string
  messages: Message[]
  draft: string | null // the streaming bubble; null = none in progress
  status: 'idle' | 'streaming'
  activity: string | null // "searching knowledge base…" style status line
  notice: string | null // transient: steered / cancelled / error
}

type Action =
  | { type: 'reset'; conversationId: string; messages?: Message[] }
  | { type: 'sent'; text: string }
  | { type: 'event'; event: TurnEvent }
  | { type: 'finished' }
  | { type: 'notice'; notice: string | null }

let seq = 0
const nextId = () => `m${++seq}`

const TOOL_LABELS: Record<string, string> = {
  kb_search: 'searching knowledge base…',
  web_search: 'searching the web…',
}

export function initialState(conversationId: string): ConversationState {
  return {
    conversationId,
    messages: [],
    draft: null,
    status: 'idle',
    activity: null,
    notice: null,
  }
}

export function reduce(state: ConversationState, action: Action): ConversationState {
  switch (action.type) {
    case 'reset':
      return { ...initialState(action.conversationId), messages: action.messages ?? [] }
    case 'sent':
      return {
        ...state,
        messages: [...state.messages, { id: nextId(), role: 'user', text: action.text }],
        draft: '',
        status: 'streaming',
        activity: null,
        notice: null,
      }
    case 'notice':
      return { ...state, notice: action.notice }
    case 'finished':
      return { ...state, status: 'idle', draft: null, activity: null }
    case 'event':
      return onEvent(state, action.event)
  }
}

function onEvent(state: ConversationState, event: TurnEvent): ConversationState {
  switch (event.event) {
    case 'text_delta':
      return { ...state, draft: (state.draft ?? '') + event.text, activity: null }
    case 'tool_started':
      return { ...state, activity: TOOL_LABELS[event.call.name] ?? `running ${event.call.name}…` }
    case 'tool_progress':
      return { ...state, activity: event.message }
    case 'tool_result_event':
      return { ...state, activity: event.result.is_error ? 'tool failed, continuing…' : 'thinking…' }
    case 'steered':
      return { ...state, notice: 'Your message was folded into the current answer.' }
    case 'warning':
      return { ...state, notice: event.message }
    case 'error':
      return { ...state, notice: `Error: ${event.message}` }
    case 'cancelled':
      return { ...state, notice: 'Stopped.' }
    case 'turn_complete':
      return { ...state, activity: null }
    case 'envelope': {
      // the accepted answer (with its References section) replaces the draft
      const { envelope } = event
      const text = envelope.answer || state.draft || ''
      const messages = text
        ? [...state.messages, { id: nextId(), role: 'assistant' as const, text, envelope }]
        : state.messages
      return { ...state, messages, draft: null }
    }
    default:
      return state
  }
}

export function useConversation(conversationId: string) {
  const [state, dispatch] = useReducer(reduce, conversationId, initialState)
  const abort = useRef<AbortController | null>(null)

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim()
      if (!trimmed) return
      if (state.status === 'streaming') {
        // mid-turn: the service steers the running turn; no second stream
        try {
          await postMessage(state.conversationId, trimmed)
          dispatch({ type: 'notice', notice: 'Your message was folded into the current answer.' })
        } catch (e) {
          dispatch({ type: 'notice', notice: `Error: ${(e as Error).message}` })
        }
        return
      }
      dispatch({ type: 'sent', text: trimmed })
      abort.current = new AbortController()
      try {
        await postMessage(
          state.conversationId,
          trimmed,
          (event) => dispatch({ type: 'event', event }),
          abort.current.signal,
        )
      } catch (e) {
        if ((e as Error).name !== 'AbortError') {
          dispatch({ type: 'notice', notice: `Error: ${(e as Error).message}` })
        }
      } finally {
        dispatch({ type: 'finished' })
      }
    },
    [state.conversationId, state.status],
  )

  const stop = useCallback(async () => {
    // EXPLICIT cancel — the only way a turn dies early (a dropped connection
    // never cancels; the kernel checkpoints and keeps the partial work)
    try {
      await cancelTurn(state.conversationId)
    } catch (e) {
      dispatch({ type: 'notice', notice: `Error: ${(e as Error).message}` })
    }
  }, [state.conversationId])

  /** Switch to a brand-new conversation: nothing to fetch, the first POST claims it. */
  const start = useCallback((id: string) => {
    abort.current?.abort()
    dispatch({ type: 'reset', conversationId: id })
  }, [])

  const load = useCallback(async (id: string) => {
    abort.current?.abort()
    const view = await getConversation(id)
    const messages: Message[] = (view?.messages ?? [])
      .filter((m) => m.role === 'user' || m.role === 'assistant')
      .map((m) => ({ id: nextId(), role: m.role as 'user' | 'assistant', text: m.text }))
    dispatch({ type: 'reset', conversationId: id, messages })
  }, [])

  return { state, send, stop, load, start }
}
