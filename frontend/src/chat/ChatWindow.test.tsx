import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { setupServer } from 'msw/node'
import App from '../App'
import { envelope, http, HttpResponse, sseResponse } from '../test/sse'

const server = setupServer(
  http.get('/v1/conversations', () => HttpResponse.json({ conversations: [] })),
)
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

const MESSAGES = '/v1/conversations/:id/messages'

async function ask(text: string) {
  const user = userEvent.setup()
  await user.type(screen.getByLabelText('message'), text)
  await user.keyboard('{Enter}')
  return user
}

test('streams text live, then the envelope replaces the draft with the accepted answer', async () => {
  server.use(
    http.post(MESSAGES, () =>
      sseResponse([
        { event: 'turn_started', data: {} },
        { event: 'text_delta', data: { text: 'the ' } },
        { event: 'text_delta', data: { text: 'answer' } },
        { event: 'turn_complete', data: { reason: 'stopped' } },
        { event: 'envelope', data: envelope('the answer [1]\n\n### References\n[1] Handbook.pdf') },
      ]),
    ),
  )
  render(<App />)
  await ask('what is my leave benefits')

  // the user's bubble is there immediately
  expect(screen.getByText('what is my leave benefits')).toBeInTheDocument()
  // the final bubble is the ENVELOPE's answer (with the References section
  // rendered as markdown), not a concat of deltas
  const heading = await screen.findByRole('heading', { name: 'References' })
  const bubble = heading.closest('[data-role="assistant"]')
  expect(bubble).not.toBeNull()
  expect(bubble).toHaveTextContent('the answer [1]')
  expect(bubble).toHaveTextContent('[1] Handbook.pdf')
  expect(screen.getByTestId('signal')).toHaveAttribute('data-signal', 'unjudged')
  expect(screen.getByTestId('signal')).toHaveTextContent('unchecked')
  // the composer is back to idle
  expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument()
})

test('a judged deployment (no deltas) renders the same way, with the signal badge', async () => {
  server.use(
    http.post(MESSAGES, () =>
      sseResponse([
        { event: 'tool_started', data: { call: { id: 't1', name: 'kb_search', arguments: '{}' } } },
        { event: 'tool_result_event', data: { result: { call_id: 't1', content: '[1] x', is_error: false } } },
        { event: 'turn_complete', data: { reason: 'stopped' } },
        { event: 'envelope', data: envelope('graded answer', { signal: 'ok', score: 0.9 }) },
      ]),
    ),
  )
  render(<App />)
  await ask('q')
  expect(await screen.findByText('graded answer')).toBeInTheDocument()
  const badge = screen.getByTestId('signal')
  expect(badge).toHaveAttribute('data-signal', 'ok')
  expect(badge).toHaveTextContent('confidence: 0.90')
})

test('a cancelled turn shows no_answer and the stop notice', async () => {
  server.use(
    http.post(MESSAGES, () =>
      sseResponse([
        { event: 'text_delta', data: { text: 'partial' } },
        { event: 'cancelled', data: {} },
        { event: 'turn_complete', data: { reason: 'cancelled' } },
        { event: 'envelope', data: envelope('', { signal: 'no_answer', stop_reason: 'cancelled' }) },
      ]),
    ),
  )
  render(<App />)
  await ask('q')
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Stopped.'))
  // the partial text is kept as the bubble (envelope had no answer)
  expect(screen.getByText('partial')).toBeInTheDocument()
  expect(screen.getByTestId('signal')).toHaveAttribute('data-signal', 'no_answer')
  expect(screen.getByTestId('signal')).toHaveTextContent('no answer')
})

test('Stop calls the cancel endpoint while a turn streams', async () => {
  let cancelled = false
  let release!: () => void
  const gate = new Promise<void>((r) => (release = r))
  server.use(
    http.post(MESSAGES, async () => {
      await gate
      return sseResponse([
        { event: 'turn_complete', data: { reason: 'cancelled' } },
        { event: 'envelope', data: envelope('', { signal: 'no_answer', stop_reason: 'cancelled' }) },
      ])
    }),
    http.post('/v1/conversations/:id/cancel', () => {
      cancelled = true
      release()
      return HttpResponse.json({ status: 'cancelling' }, { status: 202 })
    }),
  )
  render(<App />)
  const user = await ask('q')
  await user.click(await screen.findByRole('button', { name: 'Stop' }))
  await waitFor(() => expect(cancelled).toBe(true))
  await screen.findByRole('button', { name: 'Send' })
})

test('sending mid-turn steers (202) instead of opening a second stream', async () => {
  let posts = 0
  let release!: () => void
  const gate = new Promise<void>((r) => (release = r))
  server.use(
    http.post(MESSAGES, async () => {
      posts += 1
      if (posts === 2) {
        release()
        return HttpResponse.json({ status: 'steered' }, { status: 202 })
      }
      await gate
      return sseResponse([
        { event: 'turn_complete', data: { reason: 'stopped' } },
        { event: 'envelope', data: envelope('combined answer') },
      ])
    }),
  )
  render(<App />)
  const user = await ask('first')
  await screen.findByRole('button', { name: 'Stop' })
  await user.type(screen.getByLabelText('message'), 'also this')
  await user.keyboard('{Enter}')
  await waitFor(() =>
    expect(screen.getByRole('status')).toHaveTextContent('folded into the current answer'),
  )
  expect(await screen.findByText('combined answer')).toBeInTheDocument()
  expect(posts).toBe(2)
})

test('an HTTP failure surfaces as a notice and frees the composer', async () => {
  server.use(http.post(MESSAGES, () => HttpResponse.text('nope', { status: 401 })))
  render(<App />)
  await ask('q')
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Error'))
  expect(screen.getByRole('button', { name: 'Send' })).toBeInTheDocument()
})
