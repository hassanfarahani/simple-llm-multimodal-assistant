<script setup lang="ts">
import { computed, ref } from 'vue'
import EvalSection from '@/components/EvalSection.vue'
import type { AnswerSummary, CategoryScore, Level, MetricCard, RetrievalSummary } from '@/types/ragEval'

const API_URL = 'http://localhost:8000/rag-eval'

// same thresholds as evaluator.py
const level = (v: number, green: number, amber: number): Level =>
  v >= green ? 'good' : v >= amber ? 'warn' : 'bad'

async function postJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, { method: 'POST' })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail ?? `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

function barData(rows: CategoryScore[] = [], label: string) {
  return {
    labels: rows.map((r) => r.category),
    datasets: [{ label, data: rows.map((r) => r.value), backgroundColor: '#6366f1', borderRadius: 4 }],
  }
}

function barOptions(title: string, yLabel: string, min: number, max: number) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false }, title: { display: true, text: title } },
    scales: {
      x: { title: { display: true, text: 'Category' } },
      y: { min, max, title: { display: true, text: yLabel } },
    },
  }
}

// ---------- Retrieval ----------
const retrieval = ref<RetrievalSummary | null>(null)
const retrievalLoading = ref(false)
const retrievalError = ref('')

async function runRetrieval() {
  retrievalLoading.value = true
  retrievalError.value = ''
  try {
    retrieval.value = await postJson<RetrievalSummary>('/retrieval')
  } catch (e) {
    retrievalError.value = e instanceof Error ? e.message : String(e)
  } finally {
    retrievalLoading.value = false
  }
}

const retrievalCards = computed<MetricCard[]>(() => {
  const r = retrieval.value
  if (!r) return []
  return [
    { label: 'Mean Reciprocal Rank (MRR)', display: r.mrr.toFixed(4), level: level(r.mrr, 0.9, 0.75) },
    { label: 'Normalized DCG (nDCG)', display: r.ndcg.toFixed(4), level: level(r.ndcg, 0.9, 0.75) },
    { label: 'Keyword Coverage', display: `${r.keyword_coverage.toFixed(1)}%`, level: level(r.keyword_coverage, 90, 75) },
  ]
})

// ---------- Answer ----------
const answer = ref<AnswerSummary | null>(null)
const answerLoading = ref(false)
const answerError = ref('')

async function runAnswer() {
  answerLoading.value = true
  answerError.value = ''
  try {
    answer.value = await postJson<AnswerSummary>('/answer')
  } catch (e) {
    answerError.value = e instanceof Error ? e.message : String(e)
  } finally {
    answerLoading.value = false
  }
}

const answerCards = computed<MetricCard[]>(() => {
  const a = answer.value
  if (!a) return []
  const score = (label: string, v: number): MetricCard => ({ label, display: `${v.toFixed(2)}/5`, level: level(v, 4.5, 4.0) })
  return [score('Accuracy', a.accuracy), score('Completeness', a.completeness), score('Relevance', a.relevance)]
})
</script>

<template>
  <div class="mx-auto max-w-6xl px-4">
    <h1 class="text-2xl font-bold">📊 RAG Evaluation Dashboard</h1>
    <p class="mb-6 text-gray-500">Evaluate retrieval and answer quality for the Insurellm RAG system</p>

    <EvalSection
      title="🔍 Retrieval Evaluation"
      :cards="retrievalCards"
      :count="retrieval?.count ?? 0"
      :chart-data="barData(retrieval?.by_category, 'Average MRR')"
      :chart-options="barOptions('Average MRR by Category', 'Average MRR', 0, 1)"
      :loading="retrievalLoading"
      :error="retrievalError"
      @run="runRetrieval"
    />

    <EvalSection
      title="💬 Answer Evaluation"
      :cards="answerCards"
      :count="answer?.count ?? 0"
      :chart-data="barData(answer?.by_category, 'Average Accuracy')"
      :chart-options="barOptions('Average Accuracy by Category', 'Average Accuracy', 1, 5)"
      :loading="answerLoading"
      :error="answerError"
      @run="runAnswer"
    />
  </div>
</template>