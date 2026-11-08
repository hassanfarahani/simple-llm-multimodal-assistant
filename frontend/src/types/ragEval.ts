export interface CategoryScore { category: string; value: number }

export interface RetrievalSummary {
  count: number; mrr: number; ndcg: number; keyword_coverage: number
  by_category: CategoryScore[]
}

export interface AnswerSummary {
  count: number; accuracy: number; completeness: number; relevance: number
  by_category: CategoryScore[]
}

export type Level = 'good' | 'warn' | 'bad'
export interface MetricCard { label: string; display: string; level: Level }