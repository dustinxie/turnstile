import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { setupServer } from 'msw/node'
import App from '../App'
import { envelope, http, HttpResponse, sseResponse } from '../test/sse'

const LIST = [
  { conversation_id: 'c2', title: 'how many sick hours?', turn_counter: 1, in_flight: false },
  { conversation_id: 'c1', title: 'what is my leave benefits', turn_counter: 2, in_flight: true },
]
const VIEWS: Record<string, unknown> = {
  c2: {
    conversation_id: 'c2',
    turn_counter: 1,
    in_flight: false,
    messages: [
      { role: 'system', text: 'persona' },
      { role: 'user', text: 'how many sick hours?' },
      { role: 'tool', text: '[1] hrus::x.pdf#L1\nchunk' },
      {
        role: 'assistant',
        text: '80 hours [1].',
        // the persisted sidecar: verdict + references come back with history
        signal: 'ok',
        score: 0.85,
        references: [
          { n: 1, title: 'x.pdf', url: '/v1/files/tok-fresh#L1', cited: true },
          { n: 2, title: 'unused.pdf', url: null, cited: false },
        ],
      },
    ],
  },
}

const server = setupServer(
  http.get('/health', () => HttpResponse.json({ status: 'ok', spec: 's', auth: false, sso: false })),
  http.get('/v1/conversations', () => HttpResponse.json({ conversations: LIST })),
  http.get('/v1/conversations/:id', ({ params }) => {
    const view = VIEWS[params.id as string]
    return view ? HttpResponse.json(view) : HttpResponse.json({ detail: 'unknown' }, { status: 404 })
  }),
)
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

test('lists conversations newest first with the in-flight marker', async () => {
  render(<App />)
  const nav = await screen.findByRole('navigation', { name: 'conversations' })
  const names = (await screen.findAllByRole('button', { name: /sick hours|leave benefits/ })).map(
    (b) => b.textContent,
  )
  expect(names).toEqual(['how many sick hours?', 'what is my leave benefits'])
  expect(nav.querySelector('[aria-label="answering"]')).not.toBeNull() // c1 still running
})

test('selecting a conversation loads its history (user + assistant only)', async () => {
  render(<App />)
  const user = userEvent.setup()
  await user.click(await screen.findByRole('button', { name: 'how many sick hours?' }))
  expect(await screen.findByText(/80 hours/)).toBeInTheDocument()
  expect(screen.getByText('how many sick hours?', { selector: '[data-role="user"]' })).toBeInTheDocument()
  expect(screen.queryByText('persona')).toBeNull() // system/tool rows are not transcript
  // reloaded turn renders like a live one: badge + References (cited only)
  expect(screen.getByTestId('signal')).toHaveTextContent('confidence: 0.85')
  expect(screen.getByRole('heading', { name: 'References' })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'x.pdf' })).toHaveAttribute('href', '/v1/files/tok-fresh#L1')
  expect(screen.queryByText(/unused\.pdf/)).toBeNull()
  expect(screen.getByRole('button', { name: 'how many sick hours?' })).toHaveAttribute(
    'aria-current',
    'true',
  )
})

test('New chat clears the window; the list refreshes after the first turn settles', async () => {
  let listCalls = 0
  server.use(
    http.get('/v1/conversations', () => {
      listCalls += 1
      return HttpResponse.json({
        conversations:
          listCalls > 1
            ? [{ conversation_id: 'new', title: 'fresh q', turn_counter: 1, in_flight: false }, ...LIST]
            : LIST,
      })
    }),
    http.post('/v1/conversations/:id/messages', () =>
      sseResponse([
        { event: 'turn_complete', data: { reason: 'stopped' } },
        { event: 'envelope', data: envelope('fresh answer') },
      ]),
    ),
  )
  render(<App />)
  const user = userEvent.setup()
  await user.click(await screen.findByRole('button', { name: 'how many sick hours?' }))
  await screen.findByText(/80 hours/)

  await user.click(screen.getByRole('button', { name: 'New chat' }))
  expect(screen.queryByText(/80 hours/)).toBeNull()
  expect(screen.getByText('Ask a question to start a conversation.')).toBeInTheDocument()

  await user.type(screen.getByLabelText('message'), 'fresh q')
  await user.keyboard('{Enter}')
  await screen.findByText('fresh answer')
  await waitFor(() => expect(screen.getByRole('button', { name: 'fresh q' })).toBeInTheDocument())
})
