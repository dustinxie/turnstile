/**
 * The service client. One fetch wrapper attaches the Bearer token when one
 * is stored (auth on); with no token the service must be in dev mode
 * (anonymous). Streaming uses fetch + eventsource-parser because the turn
 * endpoint is a POST — native EventSource can neither POST nor set headers.
 */
import { createParser, type EventSourceMessage } from 'eventsource-parser'
import type { ConversationView, Envelope, TurnEvent } from './types'

export const TOKEN_KEY = 'turnstile.token'

export function getToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

function headers(extra: Record<string, string> = {}): HeadersInit {
  const token = getToken()
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function ok(response: Response): Promise<Response> {
  if (!response.ok) throw new ApiError(response.status, await response.text())
  return response
}

/** Decode one SSE frame into a TurnEvent (unknown names -> `other`). */
export function decodeEvent(message: EventSourceMessage): TurnEvent {
  const name = message.event ?? 'message'
  const data = message.data ? (JSON.parse(message.data) as Record<string, unknown>) : {}
  switch (name) {
    case 'turn_started':
    case 'cancelled':
      return { event: name }
    case 'text_delta':
    case 'tool_started':
    case 'tool_progress':
    case 'tool_result_event':
    case 'steered':
    case 'warning':
    case 'error':
    case 'turn_complete':
      return { event: name, ...data } as TurnEvent
    case 'envelope':
      return { event: 'envelope', envelope: data as unknown as Envelope }
    default:
      return { event: 'other', name }
  }
}

export type PostOutcome = 'streamed' | 'steered'

/**
 * POST a message; stream the turn's events to `onEvent`. Resolves 'steered'
 * (202) when a turn was already running — the text folded into it and the
 * events keep arriving on the ORIGINAL stream, so nothing else to read here.
 */
export async function postMessage(
  conversationId: string,
  text: string,
  onEvent: (event: TurnEvent) => void = () => {},
  signal?: AbortSignal,
): Promise<PostOutcome> {
  const response = await ok(
    await fetch(`/v1/conversations/${encodeURIComponent(conversationId)}/messages`, {
      method: 'POST',
      headers: headers({ 'Content-Type': 'application/json', Accept: 'text/event-stream' }),
      body: JSON.stringify({ text }),
      signal,
    }),
  )
  if (response.status === 202) return 'steered'
  if (!response.body) throw new ApiError(response.status, 'no response body')

  const parser = createParser({ onEvent: (m) => onEvent(decodeEvent(m)) })
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader()
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    parser.feed(value)
  }
  return 'streamed'
}

export async function cancelTurn(conversationId: string): Promise<'cancelling' | 'idle'> {
  const response = await ok(
    await fetch(`/v1/conversations/${encodeURIComponent(conversationId)}/cancel`, {
      method: 'POST',
      headers: headers(),
    }),
  )
  return ((await response.json()) as { status: 'cancelling' | 'idle' }).status
}

export async function getConversation(conversationId: string): Promise<ConversationView | null> {
  const response = await fetch(`/v1/conversations/${encodeURIComponent(conversationId)}`, {
    headers: headers(),
  })
  if (response.status === 404) return null
  return (await ok(response)).json() as Promise<ConversationView>
}
