import ChatbotView from '@/views/ChatbotView.vue'
import MeetingMinutesView from '@/views/MeetingMinutesView.vue'
import RagChatbotView from '@/views/RagChatbotView.vue'
import RagEvalView from '@/views/RagEvalView.vue'
import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'ChatBot',
      component: ChatbotView,
    },
    {
      path: '/meeting-minutes',
      name: 'MeetingMinutes',
      component: MeetingMinutesView,
    },
    {
      path: '/rag',
      name: 'Rag',
      component: RagChatbotView,
    },
    {
      path: '/rag-eval',
      name: 'RagEval',
      component: RagEvalView,
    }, 
  ],
})

export default router
