/**
 * Assistant text is markdown (the model writes it; the service appends the
 * "### References" section). Citations: every inline "[n]" becomes a link
 * to reference n's URL, so the body and the section point at the same place.
 *
 * File links (/v1/files/<token>) need the Bearer header, which a plain
 * browser navigation never sends — so with a token stored we fetch the file
 * ourselves and open the blob in a new tab (option 1 of the click-native
 * decision). Without a token (dev mode, anonymous) a direct link works.
 */
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Reference } from '../api/types'
import { isFileLink, linkifyCitations, openFile } from './citations'

const components: Components = {
  a: ({ href = '', children }) => {
    if (isFileLink(href)) {
      return (
        <a
          href={href}
          data-file-link
          onClick={(e) => {
            e.preventDefault()
            void openFile(href).catch((err) => console.warn(err))
          }}
        >
          {children}
        </a>
      )
    }
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    )
  },
}

export function Markdown({ text, references = [] }: { text: string; references?: Reference[] }) {
  return (
    <div className="prose prose-sm max-w-none prose-p:my-1 prose-headings:my-2 prose-ul:my-1 prose-li:my-0 prose-table:my-2 prose-th:px-2 prose-td:px-2">
      {/* GFM: tables (benefit/contribution grids), strikethrough, task lists */}
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {linkifyCitations(text, references)}
      </ReactMarkdown>
    </div>
  )
}
