// Dashboard API service for fetching real data from AstrID backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:9001'

export interface ApiResponse<T> {
  success: boolean
  data: T
  message?: string
  timestamp: string
}

export interface ObservationStats {
  total_observations: number
  processed: number
  processing: number
  failed: number
  pending: number
}

export interface DetectionStats {
  total_detections: number
  confirmed: number
  pending: number
  rejected: number
  high_confidence: number
  average_confidence: number
}

export interface WorkflowStats {
  total_workflows: number
  running: number
  completed: number
  failed: number
  pending: number
}

export interface ModelPerformance {
  model_id: string
  precision: number
  recall: number
  f1_score: number
  accuracy: number
  auc_roc: number
  confusion_matrix: number[][]
  detection_rate: number
  false_positive_rate: number
  last_evaluation: string
}

export interface WorkerHealth {
  status: 'healthy' | 'degraded' | 'unhealthy'
  active_workers: number
  total_workers: number
  healthy_workers: number
  timestamp: string
}

export interface WorkerMetrics {
  time_window_hours: number
  total_tasks_processed: number
  total_tasks_failed: number
  failure_rate: number
  average_processing_time: number
  average_memory_usage_mb: number
  average_cpu_usage_percent: number
  active_workers: number
  timestamp: string
}

export interface QueueStatus {
  queue_name: string
  worker_type: string
  priority: number
  concurrency: number
  timeout: number
  enabled: boolean
}

class DashboardApiService {
  private baseUrl: string
  private authToken: string | null = null

  constructor() {
    this.baseUrl = API_BASE_URL
    // Get auth token from localStorage or context
    if (typeof window !== 'undefined') {
      this.authToken = localStorage.getItem('auth_token')
    }
  }

  private getHeaders(): HeadersInit {
    const headers: HeadersInit = {
      'Content-Type': 'application/json',
    }

    if (this.authToken) {
      headers.Authorization = `Bearer ${this.authToken}`
    }

    return headers
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.getHeaders(),
          ...options.headers,
        },
      })

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`)
      }

      const data = await response.json()
      return data.data || data // Handle both wrapped and unwrapped responses
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error)
      throw error
    }
  }

  // Observations API
  async getObservationStats(): Promise<ObservationStats> {
    try {
      // Get total observations count
      const observations = await this.request<any[]>('/v1/observations?limit=1')

      // For now, return mock data structure that matches your training pipeline
      // In production, you'd calculate these from actual observation data
      return {
        total_observations: 1247, // This would come from actual count
        processed: 1156,
        processing: 23,
        failed: 68,
        pending: 0
      }
    } catch (error) {
      console.error('Failed to fetch observation stats:', error)
      // Return fallback data
      return {
        total_observations: 0,
        processed: 0,
        processing: 0,
        failed: 0,
        pending: 0
      }
    }
  }

  async getObservations(filters: {
    survey?: string
    status?: string
    filter?: string
    dateRange?: string
    limit?: number
    offset?: number
  } = {}) {
    const params = new URLSearchParams()

    if (filters.survey) params.append('survey', filters.survey)
    if (filters.status) params.append('status', filters.status)
    if (filters.filter) params.append('filter_band', filters.filter)
    if (filters.dateRange) params.append('date_from', filters.dateRange)
    if (filters.limit) params.append('limit', filters.limit.toString())
    if (filters.offset) params.append('offset', filters.offset.toString())

    return this.request<any[]>(`/v1/observations?${params.toString()}`)
  }

  // Detections API
  async getDetectionStats(): Promise<DetectionStats> {
    try {
      const stats = await this.request<any>('/v1/detections/statistics')

      return {
        total_detections: stats.total_detections || 0,
        confirmed: stats.validated_detections || 0,
        pending: stats.pending_validation || 0,
        rejected: stats.total_detections - stats.validated_detections - stats.pending_validation || 0,
        high_confidence: stats.high_confidence_detections || 0,
        average_confidence: stats.average_confidence || 0
      }
    } catch (error) {
      console.error('Failed to fetch detection stats:', error)
      return {
        total_detections: 0,
        confirmed: 0,
        pending: 0,
        rejected: 0,
        high_confidence: 0,
        average_confidence: 0
      }
    }
  }

  async getDetections(filters: {
    status?: string
    confidence?: string
    magnitude?: string
    dateRange?: string
    limit?: number
    offset?: number
  } = {}) {
    const params = new URLSearchParams()

    if (filters.status) params.append('status', filters.status)
    if (filters.confidence) {
      const minConfidence = filters.confidence === 'high' ? '0.8' :
                           filters.confidence === 'medium' ? '0.6' : '0.0'
      params.append('min_confidence', minConfidence)
    }
    if (filters.dateRange) params.append('date_from', filters.dateRange)
    if (filters.limit) params.append('limit', filters.limit.toString())
    if (filters.offset) params.append('offset', filters.offset.toString())

    return this.request<any[]>(`/v1/detections?${params.toString()}`)
  }

  // Model Performance API
  async getModelPerformance(modelId: string): Promise<ModelPerformance> {
    try {
      return await this.request<ModelPerformance>(`/v1/detections/models/${modelId}/performance`)
    } catch (error) {
      console.error('Failed to fetch model performance:', error)
      // Return fallback data
      return {
        model_id: modelId,
        precision: 0,
        recall: 0,
        f1_score: 0,
        accuracy: 0,
        auc_roc: 0,
        confusion_matrix: [[0, 0], [0, 0]],
        detection_rate: 0,
        false_positive_rate: 0,
        last_evaluation: new Date().toISOString()
      }
    }
  }

  // Workers API
  async getWorkerHealth(): Promise<WorkerHealth> {
    try {
      return await this.request<WorkerHealth>('/v1/workers/health')
    } catch (error) {
      console.error('Failed to fetch worker health:', error)
      return {
        status: 'unhealthy',
        active_workers: 0,
        total_workers: 0,
        healthy_workers: 0,
        timestamp: new Date().toISOString()
      }
    }
  }

  async getWorkerMetrics(timeWindowHours: number = 24): Promise<WorkerMetrics> {
    try {
      return await this.request<WorkerMetrics>(`/v1/workers/metrics?time_window_hours=${timeWindowHours}`)
    } catch (error) {
      console.error('Failed to fetch worker metrics:', error)
      return {
        time_window_hours: timeWindowHours,
        total_tasks_processed: 0,
        total_tasks_failed: 0,
        failure_rate: 0,
        average_processing_time: 0,
        average_memory_usage_mb: 0,
        average_cpu_usage_percent: 0,
        active_workers: 0,
        timestamp: new Date().toISOString()
      }
    }
  }

  async getQueueStatus(): Promise<QueueStatus[]> {
    try {
      const response = await this.request<{queues: QueueStatus[]}>('/v1/workers/queues')
      return response.queues || []
    } catch (error) {
      console.error('Failed to fetch queue status:', error)
      return []
    }
  }

  // Workflows API (placeholder - you'll need to implement these endpoints)
  async getWorkflowStats(): Promise<WorkflowStats> {
    try {
      // This would call your workflow API when implemented
      // For now, return mock data
      return {
        total_workflows: 24,
        running: 3,
        completed: 18,
        failed: 3,
        pending: 0
      }
    } catch (error) {
      console.error('Failed to fetch workflow stats:', error)
      return {
        total_workflows: 0,
        running: 0,
        completed: 0,
        failed: 0,
        pending: 0
      }
    }
  }

  async getWorkflows(filters: {
    status?: string
    type?: string
    dateRange?: string
  } = {}) {
    try {
      // This would call your workflow API when implemented
      // For now, return mock data
      return []
    } catch (error) {
      console.error('Failed to fetch workflows:', error)
      return []
    }
  }

  // MLflow Integration - Get latest model metrics
  async getLatestModelMetrics(): Promise<{
    best_val_loss: number
    final_accuracy: number
    final_f1_score: number
    training_energy_wh: number
    training_carbon_footprint_kg: number
    last_training: string
  }> {
    try {
      // This would integrate with your MLflow API or database
      // For now, return mock data based on your training notebook structure
      return {
        best_val_loss: 0.1234,
        final_accuracy: 0.95,
        final_f1_score: 0.92,
        training_energy_wh: 1250.5,
        training_carbon_footprint_kg: 0.000291,
        last_training: new Date().toISOString()
      }
    } catch (error) {
      console.error('Failed to fetch model metrics:', error)
      return {
        best_val_loss: 0,
        final_accuracy: 0,
        final_f1_score: 0,
        training_energy_wh: 0,
        training_carbon_footprint_kg: 0,
        last_training: new Date().toISOString()
      }
    }
  }
}

export const dashboardApi = new DashboardApiService()
