/**
 * Citation plumbing shared by the markdown renderer: [n] -> link rewriting,
 * and opening file links the way the auth model requires (see Markdown.tsx).
 */
import { getToken } from '../api/client'
import type { Reference } from '../api/types'

const CITE_RE = /\[(\d+)\](?!\()/g

/** Turn "[n]" into "[\[n\]](url)" for every reference that has a URL. */
export function linkifyCitations(text: string, references: Reference[]): string {
  const urls = new Map(references.filter((r) => r.url).map((r) => [r.n, r.url as string]))
  if (urls.size === 0) return text
  return text.replace(CITE_RE, (marker, n) => {
    const url = urls.get(Number(n))
    return url ? `[\\[${n}\\]](${url})` : marker
  })
}

export const isFileLink = (href: string) => href.startsWith('/v1/files/')

export async function openFile(href: string): Promise<void> {
  const token = getToken()
  if (!token) {
    window.open(href, '_blank', 'noopener')
    return
  }
  // the fragment (#L<line>) is a chunk anchor; it cannot ride a blob URL
  const response = await fetch(href.split('#', 1)[0], {
    headers: { Authorization: `Bearer ${token}` },
  })
  if (!response.ok) throw new Error(`file ${response.status}`)
  const url = URL.createObjectURL(await response.blob())
  window.open(url, '_blank', 'noopener')
}
