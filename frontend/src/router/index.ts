import ChatbotView from '@/views/ChatbotView.vue'
import MeetingMinutesView from '@/views/MeetingMinutesView.vue'
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
  ],
})

export default router
