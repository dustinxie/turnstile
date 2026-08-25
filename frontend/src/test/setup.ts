import '@testing-library/jest-dom/vitest'

// jsdom has no layout: scrollIntoView is missing. The chat window scrolls to
// the newest bubble on every update; make that a no-op under test.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
