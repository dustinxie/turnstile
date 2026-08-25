/** Test helpers: build an SSE body the way the service frames it. */
import { http, HttpResponse } from 'msw'

export function sseBody(frames: { event: string; data: unknown }[]): string {
  return frames.map((f) => `event: ${f.event}\ndata: ${JSON.stringify(f.data)}\n\n`).join('')
}

export function sseResponse(frames: { event: string; data: unknown }[]) {
  return new HttpResponse(sseBody(frames), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream' },
  })
}

export const envelope = (answer: string, extra: Record<string, unknown> = {}) => ({
  conversation_id: 'c1',
  answer,
  signal: 'unjudged',
  score: null,
  references: [],
  stop_reason: 'stopped',
  ...extra,
})

export { http, HttpResponse }
