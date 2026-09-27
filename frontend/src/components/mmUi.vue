<script setup lang="ts">
import { ref, computed } from 'vue'
import Button from 'primevue/button'
import FileUpload, { type FileUploadSelectEvent } from 'primevue/fileupload'
import ProgressBar from 'primevue/progressbar'
import Message from 'primevue/message'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

const props = withDefaults(
  defineProps<{
    endpoint?: string
    fieldName?: string
    maxSizeMB?: number
  }>(),
  {
    endpoint: 'http://localhost:8000/audio/upload',
    fieldName: 'file',
    maxSizeMB: 500,
  }
)

const emit = defineEmits<{
  (e: 'uploaded', response: any): void
  (e: 'error', error: any): void
}>()

const file = ref<File | null>(null)
const uploading = ref(false)
const progress = ref(0)
const errorMsg = ref<string | null>(null)
const successMsg = ref<string | null>(null)
const result = ref<string | null>(null)

const maxBytes = computed(() => props.maxSizeMB * 1024 * 1024)

const renderedSummary = computed(() => {
  if (!result.value) return ''
  const raw = marked.parse(result.value, { async: false }) as string
  return DOMPurify.sanitize(raw)
})

function onSelect(event: FileUploadSelectEvent) {
  errorMsg.value = null
  successMsg.value = null
  result.value = null

  const selected = event.files?.[0]
  if (!selected) return

  if (!selected.type.startsWith('audio/')) {
    errorMsg.value = 'Please select a valid audio file.'
    file.value = null
    return
  }
  if (selected.size > maxBytes.value) {
    errorMsg.value = `File is too large. Max ${props.maxSizeMB} MB.`
    file.value = null
    return
  }
  file.value = selected
}

function onRemove() {
  file.value = null
  result.value = null
  successMsg.value = null
  errorMsg.value = null
}

async function upload() {
  if (!file.value) {
    errorMsg.value = 'Please choose an audio file first.'
    return
  }

  uploading.value = true
  progress.value = 0
  errorMsg.value = null
  successMsg.value = null
  result.value = null

  const formData = new FormData()
  formData.append(props.fieldName, file.value)

  try {
    // Use XHR for upload progress; fetch() doesn't support it cleanly.
    const response = await new Promise<any>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', props.endpoint)

      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          progress.value = Math.round((e.loaded / e.total) * 100)
        }
      }

      xhr.onload = () => {
        const isJson =
          xhr.getResponseHeader('content-type')?.includes('application/json')
        const body = isJson ? JSON.parse(xhr.responseText) : xhr.responseText

        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(body)
        } else {
          reject({ status: xhr.status, body })
        }
      }

      xhr.onerror = () => reject(new Error('Network error'))
      xhr.send(formData)
    })
    console.log('response ........', response)

    result.value = response.summary
    console.log('response ........', result.value)
    successMsg.value = 'Audio uploaded successfully!'
    emit('uploaded', response)
  } catch (err: any) {
    console.error(err)
    errorMsg.value =
      err?.body?.detail || err?.message || 'Upload failed. Please try again.'
    emit('error', err)
  } finally {
    uploading.value = false
  }
}
</script>

<template>
  <div class="mx-auto w-full max-w-xl rounded-2xl bg-white p-6 shadow-lg">
    <h2 class="mb-4 text-xl font-semibold text-gray-800">Upload Audio</h2>

    <FileUpload
      mode="advanced"
      :auto="false"
      :custom-upload="true"
      accept="audio/*"
      :max-file-size="maxBytes"
      :show-upload-button="false"
      :show-cancel-button="false"
      :multiple="false"
      choose-label="Choose Audio"
      :disabled="uploading"
      @select="onSelect"
      @remove="onRemove"
      @clear="onRemove"
    >
      <template #empty>
        <div class="flex flex-col items-center justify-center py-10 text-gray-500">
          <i class="pi pi-microphone mb-3 text-4xl text-indigo-500" />
          <p class="text-sm">Drag and drop an audio file here, or click Choose.</p>
        </div>
      </template>
    </FileUpload>

    <div v-if="uploading || progress > 0" class="mt-4">
      <ProgressBar :value="progress" />
      <p class="mt-1 text-right text-xs text-gray-500">{{ progress }}%</p>
    </div>

    <div class="mt-4 flex items-center gap-3">
      <Button
        label="Upload"
        icon="pi pi-upload"
        :loading="uploading"
        :disabled="!file || uploading"
        @click="upload"
      />
      <Button
        v-if="file && !uploading"
        label="Clear"
        icon="pi pi-times"
        severity="secondary"
        text
        @click="onRemove"
      />
      <span v-if="file" class="text-sm text-gray-600">{{ file.name }}</span>
    </div>

    <Message v-if="errorMsg" severity="error" :closable="false" class="mt-4">
      {{ errorMsg }}
    </Message>

    <Message v-if="successMsg" severity="success" :closable="false" class="mt-4">
      {{ successMsg }}
    </Message>

    <article
      v-if="renderedSummary"
      class="prose prose-sm mt-4 max-h-[32rem] overflow-auto rounded-lg bg-gray-50 p-4 text-gray-800"
      v-html="renderedSummary"
    />
  </div>
</template>