<script setup lang="ts">
import Button from 'primevue/button'
import Chart from 'primevue/chart'
import type { Level, MetricCard } from '@/types/ragEval'

defineProps<{
  title: string
  cards: MetricCard[]
  count: number
  chartData: object
  chartOptions: object
  loading: boolean
  error: string
}>()
const emit = defineEmits<{ run: [] }>()

const LEVEL_CLASSES: Record<Level, string> = {
  good: 'border-green-600 text-green-600',
  warn: 'border-amber-500 text-amber-500',
  bad: 'border-red-600 text-red-600',
}
</script>

<template>
  <section class="mb-10">
    <h2 class="mb-3 text-xl font-semibold">{{ title }}</h2>
    <Button label="Run Evaluation" class="w-full" :loading="loading" @click="emit('run')" />

    <div class="mt-4 grid gap-4 md:grid-cols-2">
      <div>
        <p v-if="loading" class="p-5 text-center text-gray-400">Running all tests – this can take a few minutes…</p>
        <p v-else-if="error" class="rounded bg-red-100 p-3 text-red-700">{{ error }}</p>
        <p v-else-if="!cards.length" class="p-5 text-center text-gray-400">Click 'Run Evaluation' to start</p>
        <template v-else>
          <div
            v-for="card in cards"
            :key="card.label"
            class="my-2 rounded-lg border-l-4 bg-gray-100 p-4 dark:bg-gray-800"
            :class="LEVEL_CLASSES[card.level]"
          >
            <div class="mb-1 text-sm text-gray-500">{{ card.label }}</div>
            <div class="text-3xl font-bold">{{ card.display }}</div>
          </div>
          <div class="mt-5 rounded border border-green-300 bg-green-100 p-2 text-center text-sm font-bold text-green-800">
            ✓ Evaluation Complete: {{ count }} tests
          </div>
        </template>
      </div>

      <div class="h-[400px] rounded-lg bg-gray-50 p-3 dark:bg-gray-900">
        <Chart type="bar" :data="chartData" :options="chartOptions" class="h-full" />
      </div>
    </div>
  </section>
</template>