import { render, screen } from '@testing-library/react'
import App from './App'

test('renders the two-column shell', () => {
  render(<App />)
  expect(screen.getByRole('complementary', { name: /conversation history/i })).toBeInTheDocument()
  expect(screen.getByRole('main', { name: /conversation/i })).toBeInTheDocument()
})
