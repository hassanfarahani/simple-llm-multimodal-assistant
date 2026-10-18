
<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

type ChatRole = 'user' | 'assistant'

interface ChatMessage {
  id: number
  role: ChatRole
  content: string
  time: Date
}

interface ApiMessage {
  role: 'user' | 'assistant'
  content: string
}

interface ChatApiResponse {
  // role: 'assistant'
  response: string
}

const API_URL = 'http://localhost:8000/rag-messages'

const input = ref('')
const loading = ref(false)
const error = ref('')
const messagesContainer = ref<HTMLElement | null>(null)

let nextId = 1

const messages = ref<ChatMessage[]>([
  {
    id: nextId++,
    role: 'assistant',
    content: 'Hello! 👋 How can I help you today?',
    time: new Date(),
  },
])

const canSend = computed(() => input.value.trim().length > 0 && !loading.value)

function formatTime(date: Date): string {
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
  })
}

function renderMarkdown(content: string): string {
  const html = marked.parse(content, { async: false })
  return DOMPurify.sanitize(html)
}


function sendMessageFromLastUser() {
  const lastUserMessage = [...messages.value]
    .reverse()
    .find((message) => message.role === 'user')

  if (!lastUserMessage || loading.value) return

  input.value = lastUserMessage.content
  void sendMessage()
}


async function scrollToBottom() {
  await nextTick()

  if (messagesContainer.value) {
    messagesContainer.value.scrollTop =
      messagesContainer.value.scrollHeight
  }
}

watch(
  () => messages.value.length,
  () => {
    void scrollToBottom()
  },
)


async function sendMessage() {
  const content = input.value.trim()

  if (!content || loading.value) return

  // Add the new user message to the UI history.
  messages.value.push({
    id: nextId++,
    role: 'user',
    content,
    time: new Date(),
  })

  input.value = ''
  error.value = ''
  loading.value = true

  await scrollToBottom()

  try {
    const conversation: ApiMessage[] = messages.value.map(
      ({ role, content }) => ({
        role,
        content,
      }),
    )

    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content,
        history: conversation,
      }),
    })

    if (!response.ok) {
      throw new Error(`Request failed (${response.status})`)
    }

    const data = (await response.json()) as ChatApiResponse

    // Add the LLM response to the history so it is sent
    // with the next user question.
    messages.value.push({
      id: nextId++,
      role: 'assistant',
      content: data.response,
      time: new Date(),
    })
  } catch (err) {
    error.value =
      err instanceof Error
        ? err.message
        : 'Something went wrong. Please try again.'
  } finally {
    loading.value = false
    await scrollToBottom()
  }
}

</script>

<template>
  <section
    class="mx-auto flex h-[min(1000px,85vh)] min-h-[400px] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-lg dark:border-surface-700 dark:bg-surface-900"
  >
    <!-- Header -->
    <header
      class="flex items-center gap-3 border-b border-gray-200 px-5 py-4 dark:border-surface-700"
    >
      <div
        class="flex h-11 w-11 items-center justify-center rounded-full bg-blue-600 text-xl text-white"
        aria-hidden="true"
      >
        ✦
      </div>

      <div class="min-w-0 flex-1">
        <h2 class="font-semibold text-gray-900 dark:text-surface-0">
          Chat Assistant
        </h2>
        <p class="text-sm text-gray-500">
          Ask me anything
        </p>
      </div>

      <span class="flex items-center gap-2 text-sm text-gary-500">
        <span
          class="h-2 w-2 rounded-full"
          :class="loading ? 'bg-amber-500' : 'bg-green-500'"
        />
        {{ loading ? 'Thinking...' : 'Online' }}
      </span>
    </header>

    <!-- Messages -->
    <div
      ref="messagesContainer"
      class="flex-1 space-y-5 overflow-y-auto bg-gray-50 p-4 sm:p-6 dark:bg-surface-950/40"
      role="log"
      aria-label="Chat messages"
      aria-live="polite"
    >
      <div
        v-for="message in messages"
        :key="message.id"
        class="flex items-end gap-2 sm:gap-3"
        :class="message.role === 'user' ? 'justify-end' : 'justify-start'"
      >
        <!-- Assistant avatar -->
        <div
          v-if="message.role === 'assistant'"
          class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-600 text-sm text-white"
          aria-hidden="true"
        >
          ✦
        </div>

        <div
          class="max-w-[85%] sm:max-w-[75%]"
          :class="message.role === 'user' ? 'order-first' : ''"
        >
          <div
            v-if="message.role === 'assistant'"
            class="chat-markdown rounded-2xl rounded-bl-sm bg-white px-4 py-3 text-gray-800 shadow-sm ring-1 ring-surface-200 dark:bg-surface-800 dark:text-surface-100 dark:ring-surface-700"
            v-html="renderMarkdown(message.content)"
          />

          <div
            v-else
            class="whitespace-pre-wrap break-words rounded-2xl rounded-br-sm bg-blue-600 px-4 py-3 text-white"
          >
            {{ message.content }}
          </div>

          <p
            class="mt-1 px-1 text-xs text-gray-400"
            :class="message.role === 'user' ? 'text-right' : 'text-left'"
          >
            {{ formatTime(message.time) }}
          </p>
        </div>

        <!-- User avatar -->
        <div
          v-if="message.role === 'user'"
          class="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gray-200 text-sm text-gray-700 dark:bg-surface-700 dark:text-surface-100"
          aria-hidden="true"
        >
          You
        </div>
      </div>

      <!-- Loading indicator -->
      <div v-if="loading" class="flex items-end gap-3">
        <div
          class="flex h-8 w-8 items-center justify-center rounded-full bg-blue-600 text-white"
          aria-hidden="true"
        >
          ✦
        </div>

        <div
          class="flex items-center gap-1 rounded-2xl rounded-bl-sm bg-white px-4 py-4 shadow-sm ring-1 ring-surface-200 dark:bg-surface-800 dark:ring-surface-700"
          aria-label="Assistant is typing"
        >
          <span class="typing-dot" />
          <span class="typing-dot" />
          <span class="typing-dot" />
        </div>
      </div>
    </div>

    <!-- Error -->
    <div
      v-if="error"
      class="flex items-center justify-between gap-3 border-t border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/30 dark:text-red-300"
      role="alert"
    >
      <span>{{ error }}</span>
      <button
        type="button"
        class="shrink-0 font-semibold underline"
        @click="sendMessageFromLastUser"
      >
        Retry
      </button>
    </div>

    <!-- Composer -->
    <form
      class="border-t border-gray-200 bg-white p-3 sm:p-4 dark:border-surface-700 dark:bg-surface-900"
      @submit.prevent="sendMessage"
    >
      <div
        class="flex items-end gap-2 rounded-2xl border border-gray-300 bg-surface-50 p-2 focus-within:border-blue-500 focus-within:ring-2 focus-within:ring-blue-500/20 dark:border-surface-600 dark:bg-surface-800"
      >
        <textarea
          v-model="input"
          rows="1"
          maxlength="10000"
          placeholder="Type your message..."
          aria-label="Your message"
          class="max-h-36 min-h-10 min-w-0 flex-1 resize-y bg-transparent px-2 py-2 text-sm text-surface-900 outline-none placeholder:text-gray-400 dark:text-surface-0"
          :disabled="loading"
          @keydown.enter.exact.prevent="sendMessage"
        />

        <button
          type="submit"
          :disabled="!canSend"
          class="flex h-10 shrink-0 items-center gap-2 rounded-xl bg-blue-600 px-4 text-sm font-medium text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <span aria-hidden="true">➤</span>
          <span class="hidden sm:inline">Send</span>
        </button>
      </div>

      <p class="mt-2 text-center text-xs text-gray-400">
        Enter to send · Shift + Enter for a new line
      </p>
    </form>
  </section>
</template>

<style scoped>
.typing-dot {
  width: 6px;
  height: 6px;
  border-radius: 9999px;
  background: #94a3b8;
  animation: typing 1.2s infinite ease-in-out;
}

.typing-dot:nth-child(2) {
  animation-delay: 0.15s;
}

.typing-dot:nth-child(3) {
  animation-delay: 0.3s;
}

@keyframes typing {
  0%, 60%, 100% {
    opacity: 0.4;
    transform: translateY(0);
  }

  30% {
    opacity: 1;
    transform: translateY(-4px);
  }
}

.chat-markdown :deep(p) {
  margin: 0.4rem 0;
  overflow-wrap: anywhere;
}

.chat-markdown :deep(p:first-child) {
  margin-top: 0;
}

.chat-markdown :deep(p:last-child) {
  margin-bottom: 0;
}

.chat-markdown :deep(pre) {
  max-width: 100%;
  overflow-x: auto;
  border-radius: 0.5rem;
  background: #0f172a;
  color: #e2e8f0;
  padding: 0.75rem;
  margin: 0.75rem 0;
}

.chat-markdown :deep(code) {
  overflow-wrap: anywhere;
  font-family: ui-monospace, monospace;
  font-size: 0.875em;
}

.chat-markdown :deep(pre code) {
  overflow-wrap: normal;
}

.chat-markdown :deep(ul) {
  list-style: disc;
  padding-left: 1.25rem;
  margin: 0.5rem 0;
}

.chat-markdown :deep(ol) {
  list-style: decimal;
  padding-left: 1.25rem;
  margin: 0.5rem 0;
}

.chat-markdown :deep(a) {
  color: #3b82f6;
  text-decoration: underline;
  overflow-wrap: anywhere;
}

.chat-markdown :deep(h1),
.chat-markdown :deep(h2),
.chat-markdown :deep(h3) {
  font-weight: 600;
  margin: 0.75rem 0 0.4rem;
}

.chat-markdown :deep(blockquote) {
  border-left: 3px solid #94a3b8;
  padding-left: 0.75rem;
  margin: 0.5rem 0;
}
</style>
