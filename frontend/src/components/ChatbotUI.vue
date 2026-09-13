<template>
  <div class="max-w-7xl mx-auto p-4">
    <!-- Top Row: Chatbot and Image -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
      <!-- Chatbot Panel -->
      <Card class="h-[500px]">
        <template #title>
          <div class="flex items-center gap-2">
            <i class="pi pi-comments text-2xl"></i>
            <span>Chat</span>
          </div>
        </template>
        <template #content>
          <ScrollPanel class="h-[400px] w-full pr-4">
            <div
              v-for="(message, index) in messages"
              :key="index"
              class="mb-3"
            >
              <div
                v-if="message.role === 'user'"
                class="flex justify-end"
              >
                <Chip
                  :label="message.content"
                  class="bg-blue-500 text-white px-4 py-2 max-w-[80%]"
                  style="border-radius: 1rem; height: auto; white-space: normal;"
                >
                  <template #icon>
                    <i class="pi pi-user mr-2"></i>
                  </template>
                </Chip>
              </div>
              <div v-else class="flex justify-start">
                <Chip
                  :label="message.content"
                  class="bg-gray-100 text-gray-800 px-4 py-2 max-w-[80%]"
                  style="border-radius: 1rem; height: auto; white-space: normal;"
                >
                  <template #icon>
                    <Avatar
                      icon="pi pi-android"
                      class="mr-2"
                      style="background-color: #4caf50; color: white"
                      shape="circle"
                      size="small"
                    />
                  </template>
                </Chip>
              </div>
            </div>
            <div v-if="isTyping" class="flex justify-start">
              <Chip class="bg-gray-100">
                <ProgressSpinner
                  style="width: 20px; height: 20px"
                  strokeWidth="4"
                  animationDuration="1s"
                />
                <span class="ml-2">AI is typing...</span>
              </Chip>
            </div>
          </ScrollPanel>
        </template>
      </Card>

      <!-- Image Output Panel -->
      <Card class="h-[500px]">
        <template #title>
          <div class="flex items-center gap-2">
            <i class="pi pi-image text-2xl"></i>
            <span>Image Output</span>
          </div>
        </template>
        <template #content>
          <div class="flex items-center justify-center h-[400px]">
            <Image
              v-if="currentImage"
              :src="currentImage"
              alt="AI Generated Image"
              preview
              class="max-h-full max-w-full"
            />
            <div v-else class="text-center text-gray-400">
              <i class="pi pi-image" style="font-size: 4rem"></i>
              <p class="mt-4 text-lg">No image generated yet</p>
            </div>
          </div>
        </template>
      </Card>
    </div>

    <!-- Audio Output Row -->
    <Card class="mb-4">
      <template #title>
        <div class="flex items-center gap-2">
          <i class="pi pi-volume-up text-2xl"></i>
          <span>Audio Output</span>
        </div>
      </template>
      <template #content>
        <div v-if="currentAudio">
          <audio
            :src="currentAudio"
            controls
            autoplay
            class="w-full"
          ></audio>
        </div>
        <div v-else class="text-center text-gray-400 py-4">
          <i class="pi pi-volume-off text-3xl"></i>
          <p class="mt-2">No audio available</p>
        </div>
      </template>
    </Card>

    <!-- Message Input Row -->
    <Card>
      <template #content>
        <div class="flex gap-2">
          <span class="p-input-icon-left flex-1">
            <i class="pi pi-send" />
            <InputText
              v-model="userMessage"
              placeholder="Chat with our AI Assistant..."
              class="w-full"
              @keyup.enter="sendMessage"
              :disabled="isTyping"
            />
          </span>
          <Button
            icon="pi pi-send"
            label="Send"
            @click="sendMessage"
            :disabled="!userMessage.trim() || isTyping"
            :loading="isTyping"
          />
          <Button
            icon="pi pi-trash"
            severity="danger"
            outlined
            @click="clearChat"
            v-tooltip.top="'Clear chat'"
          />
        </div>
      </template>
    </Card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Card from 'primevue/card'
import InputText from 'primevue/inputtext'
import Button from 'primevue/button'
import ScrollPanel from 'primevue/scrollpanel'
import Image from 'primevue/image'
import Chip from 'primevue/chip'
import Avatar from 'primevue/avatar'
import ProgressSpinner from 'primevue/progressspinner'

const messages = ref([
  // {
  //   role: 'assistant',
  //   content: 'Hello! I am your AI assistant. How can I help you today?'
  // } 
])
const userMessage = ref('')
const currentImage = ref(null)
const currentAudio = ref(null)
const isTyping = ref(false)

const sendMessage = async () => {
  if (!userMessage.value.trim()) return

  // Add user message
  messages.value.push({
    role: 'user',
    content: userMessage.value
  })

  const messageText = userMessage.value
  userMessage.value = ''
  isTyping.value = true
  const requestBody = {
    content: messageText,
    history: messages.value.slice(0, -1)
  }

  try {
    const response = await fetch('http://localhost:8000/message', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(requestBody)
     })
     const result = await response.json()
     console.log('res ...........', result)

    messages.value.push({
      role: 'assistant',
      content: result.model_response
    })

    if (result.destination_city_image) {
      currentImage.value = result.destination_city_image
    }

    if (result.audio_base64) {
      // Convert Base64 to raw binary data
      const binaryString = window.atob(result.audio_base64)
      const len = binaryString.length
      const bytes = new Uint8Array(len)
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i)
      }
      
      // Create a Blob and a local URL for the audio element
      const audioBlob = new Blob([bytes], { type: 'audio/mpeg' })
      // Revoke previous URL to free memory if it exists
      if (currentAudio.value) {
        URL.revokeObjectURL(currentAudio.value)
      }
      currentAudio.value = URL.createObjectURL(audioBlob)
    }

    isTyping.value = false
  } catch (error) {
    console.error('Error sending message:', error)
    messages.value.push({
      role: 'assistant',
      content: 'Sorry, there was an error processing your message. Please try again.'
    })
    isTyping.value = false
  }
}

const clearChat = () => {
  if (currentAudio.value) {
    URL.revokeObjectURL(currentAudio.value)
  }
  messages.value = [
    {
      role: 'assistant',
      content: 'Hello! I am your AI assistant. How can I help you today?'
    }
  ]
  currentImage.value = null
  currentAudio.value = null
}
</script>

<style scoped>
/* Custom scrollbar for ScrollPanel */
:deep(.p-scrollpanel-content) {
  padding-right: 1rem;
}

/* Make chips wrap text properly */
:deep(.p-chip .p-chip-text) {
  white-space: normal;
  word-wrap: break-word;
}
</style>