/**
 * What an unauthenticated visitor sees on a deployment that requires login:
 * the illustration on the left (the one employees already know from the
 * previous assistant), one action on the right. The disclaimer wording is
 * the HR-mandated one the email assistant also carries.
 */
import loginPic from '../assets/login-pic.svg'

interface Props {
  onLogin: () => void
}

export function LoginScreen({ onLogin }: Props) {
  return (
    <main aria-label="login" className="grid h-full bg-[#fcfdfe] text-neutral-900 md:grid-cols-[55%_45%]">
      <div className="hidden overflow-hidden md:block">
        <img src={loginPic} alt="" className="h-full w-full object-cover" />
      </div>
      <div className="flex flex-col items-center justify-center gap-6 px-8">
        <div className="text-center">
          <h1 className="text-4xl font-semibold">HR Assistant</h1>
          <p className="mt-3 max-w-md text-base text-neutral-600">
            Ask about benefits, leave, payroll, and policies. Answers come from the internal HR
            knowledge base with references you can open.
          </p>
        </div>
        <button
          type="button"
          onClick={onLogin}
          className="rounded-lg bg-blue-600 px-6 py-3 text-base font-medium text-white hover:bg-blue-500"
        >
          Log in as Employee
        </button>
        <p className="max-w-lg text-center text-sm text-neutral-500">
          You are interacting with an AI-powered virtual assistant, not a human agent. This
          chatbot uses artificial intelligence to provide answers to common employee questions.
        </p>
      </div>
    </main>
  )
}
