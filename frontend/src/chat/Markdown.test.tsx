import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { TOKEN_KEY } from '../api/client'
import type { Reference } from '../api/types'
import { linkifyCitations } from './citations'
import { Markdown } from './Markdown'

const REFS: Reference[] = [
  { n: 1, title: 'Handbook.pdf', url: '/v1/files/tok1#L755', cited: true },
  { n: 2, title: 'Time Off', url: 'https://sharepoint.example/time-off', cited: true },
  { n: 3, title: 'Unlinked.pdf', url: null, cited: false },
]

afterEach(() => {
  localStorage.clear()
  vi.restoreAllMocks()
})

test('renders markdown: bold, headings, lists', () => {
  render(<Markdown text={'**Vacation (Flexible Policy)**\n\n### References\n- item'} />)
  expect(screen.getByText('Vacation (Flexible Policy)').tagName).toBe('STRONG')
  expect(screen.getByRole('heading', { level: 3 })).toHaveTextContent('References')
  expect(screen.getByRole('listitem')).toHaveTextContent('item')
})

test('renders GFM tables (contribution grids)', () => {
  const table =
    '2026 monthly contributions [1]\n\n| Plan | Employee Only | +Spouse |\n|---|---|---|\n| PPO | $50 | $120 |'
  render(<Markdown text={table} />)
  expect(screen.getByRole('table')).toBeInTheDocument()
  expect(screen.getByRole('columnheader', { name: '+Spouse' })).toBeInTheDocument()
  expect(screen.getByRole('cell', { name: '$120' })).toBeInTheDocument()
})

test('inline [n] markers link to the matching reference; unknown/unlinked stay text', () => {
  expect(linkifyCitations('see [1] and [2], not [3] or [9]', REFS)).toBe(
    'see [\\[1\\]](/v1/files/tok1#L755) and [\\[2\\]](https://sharepoint.example/time-off), not [3] or [9]',
  )
  // an existing markdown link "[1](url)" is left alone (the References section shape)
  expect(linkifyCitations('[1] [Handbook.pdf](/v1/files/tok1#L755)', REFS)).toBe(
    '[\\[1\\]](/v1/files/tok1#L755) [Handbook.pdf](/v1/files/tok1#L755)',
  )
  render(<Markdown text="Vacation accrues [1]." references={REFS} />)
  const link = screen.getByRole('link', { name: '[1]' })
  expect(link).toHaveAttribute('href', '/v1/files/tok1#L755')
})

test('web links open in a new tab directly', () => {
  render(<Markdown text="Policy [2]." references={REFS} />)
  const link = screen.getByRole('link', { name: '[2]' })
  expect(link).toHaveAttribute('target', '_blank')
  expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'))
})

test('file links fetch with the Bearer token and open the blob', async () => {
  localStorage.setItem(TOKEN_KEY, 'jwt-123')
  // a minimal fetch Response shape: mixing Node's Response with jsdom's Blob
  // is version-dependent (Blob.stream missing), so fake only what openFile uses
  const pdf = new Blob(['%PDF'], { type: 'application/pdf' })
  const fetchMock = vi
    .spyOn(globalThis, 'fetch')
    .mockResolvedValue({ ok: true, status: 200, blob: async () => pdf } as unknown as Response)
  const open = vi.spyOn(window, 'open').mockImplementation(() => null)
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:pdf')

  render(<Markdown text="[1] [Handbook.pdf](/v1/files/tok1#L755)" references={REFS} />)
  fireEvent.click(screen.getByRole('link', { name: 'Handbook.pdf' }))

  await waitFor(() => expect(open).toHaveBeenCalledWith('blob:pdf', '_blank', 'noopener'))
  const [url, init] = fetchMock.mock.calls[0]
  expect(url).toBe('/v1/files/tok1') // fragment dropped: a chunk anchor cannot ride a blob URL
  expect((init as RequestInit).headers).toEqual({ Authorization: 'Bearer jwt-123' })
})

test('without a token (dev mode) file links open directly', async () => {
  const open = vi.spyOn(window, 'open').mockImplementation(() => null)
  const fetchMock = vi.spyOn(globalThis, 'fetch')
  render(<Markdown text="[1] [Handbook.pdf](/v1/files/tok1#L755)" references={REFS} />)
  fireEvent.click(screen.getByRole('link', { name: 'Handbook.pdf' }))
  await waitFor(() => expect(open).toHaveBeenCalledWith('/v1/files/tok1#L755', '_blank', 'noopener'))
  expect(fetchMock).not.toHaveBeenCalled()
})
