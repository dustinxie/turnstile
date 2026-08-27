/**
 * The service's wire contract as the frontend sees it. SSE events mirror the
 * kernel's AgentEvent DTOs verbatim (event name = snake_cased class, data =
 * its fields) — a tagged union, so one `switch` on `event` handles a turn.
 * Only the events the UI reacts to are typed; the rest are `Other`.
 */

export type Signal = 'ok' | 'low_quality' | 'unjudged' | 'no_answer'

export interface Reference {
  n: number
  title: string
  url: string | null
  cited: boolean
}

export interface Envelope {
  conversation_id: string
  answer: string
  signal: Signal
  score: number | null
  references: Reference[]
  stop_reason: string
}

export type TurnEvent =
  | { event: 'turn_started' }
  | { event: 'text_delta'; text: string }
  | { event: 'tool_started'; call: { id: string; name: string; arguments: string } }
  | { event: 'tool_progress'; call_id: string; message: string }
  | { event: 'tool_result_event'; result: { call_id: string; content: string; is_error: boolean } }
  | { event: 'steered'; count: number }
  | { event: 'warning'; message: string }
  | { event: 'error'; message: string }
  | { event: 'cancelled' }
  | { event: 'turn_complete'; reason: string }
  | { event: 'envelope'; envelope: Envelope }
  | { event: 'other'; name: string }

export interface ConversationSummary {
  conversation_id: string
  title: string // first user message, truncated server-side; '' while the first turn runs
  turn_counter: number
  in_flight: boolean
}

export interface ViewMessage {
  role: 'user' | 'assistant' | 'system' | 'tool'
  text: string
  // present on a turn's final assistant message: the persisted sidecar
  signal?: Signal
  score?: number | null
  references?: Reference[]
}

export interface ConversationView {
  conversation_id: string
  turn_counter: number
  in_flight: boolean
  messages: ViewMessage[]
}
