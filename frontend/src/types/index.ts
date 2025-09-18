// Core domain types based on ASTR-73 and ASTR-87 requirements

export interface Observation {
  id: string
  survey: string
  ra: number
  dec: number
  filter: string
  exposureTime: number
  status: 'pending' | 'processing' | 'processed' | 'failed'
  quality?: number
  timestamp: string
  processingTime?: number
  metadata: {
    airmass?: number
    seeing?: number
    skyBrightness?: number
    moonPhase?: number
  }
  wcs?: {
    crval1: number
    crval2: number
    crpix1: number
    crpix2: number
    cdelt1: number
    cdelt2: number
    crota2: number
  }
  files: {
    raw: string
    calibrated?: string
    aligned?: string
  }
}

export interface Survey {
  id: string
  name: string
  description: string
  capabilities: string[]
  filters: string[]
  status: 'active' | 'inactive' | 'maintenance'
  configuration: {
    apiEndpoint: string
    rateLimit: number
    timeout: number
  }
  statistics: {
    totalObservations: number
    processedObservations: number
    averageQuality: number
  }
}

export interface Detection {
  id: string
  observationId: string
  ra: number
  dec: number
  confidence: number
  magnitude: number
  status: 'confirmed' | 'pending' | 'rejected' | 'false_positive'
  timestamp: string
  imageUrl: string
  annotations: Annotation[]
  metadata: {
    algorithm: string
    parameters: Record<string, any>
    quality: number
  }
  classification?: {
    type: string
    probability: number
    features: Record<string, number>
  }
}

export interface Annotation {
  x: number
  y: number
  type: 'source' | 'reference' | 'artifact'
  label: string
  properties?: Record<string, any>
}

export interface Workflow {
  id: string
  name: string
  type: 'processing' | 'analysis' | 'export' | 'training'
  status: 'running' | 'completed' | 'failed' | 'pending' | 'paused'
  progress: number
  startTime?: string
  endTime?: string
  duration?: string
  tasks: {
    total: number
    completed: number
    failed: number
    pending: number
  }
  resources: {
    cpu: number
    memory: number
    storage: number
  }
  error?: string
  logs?: string[]
  parameters: Record<string, any>
}

export interface User {
  id: string
  email: string
  name: string
  role: 'admin' | 'researcher' | 'curator' | 'viewer'
  status: 'active' | 'inactive' | 'suspended'
  lastActive: string
  joinDate: string
  permissions: string[]
  preferences: {
    theme: 'dark' | 'light' | 'auto'
    timezone: string
    dateFormat: string
    notifications: {
      email: boolean
      push: boolean
      weekly: boolean
    }
  }
  activity: {
    observations: number
    detections: number
    workflows: number
  }
}

export interface SystemMetrics {
  observations: {
    total: number
    processed: number
    processing: number
    failed: number
  }
  detections: {
    total: number
    confirmed: number
    pending: number
    rejected: number
  }
  workflows: {
    total: number
    running: number
    completed: number
    failed: number
  }
  system: {
    cpu: number
    memory: number
    storage: number
    uptime: number
  }
}

// API Response types
export interface ApiResponse<T> {
  data: T
  message?: string
  status: 'success' | 'error'
  timestamp: string
}

export interface PaginatedResponse<T> {
  data: T[]
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
  }
}

// Filter types
export interface ObservationFilters {
  survey?: string
  status?: string
  filter?: string
  dateRange?: string
  qualityMin?: number
  qualityMax?: number
  raMin?: number
  raMax?: number
  decMin?: number
  decMax?: number
}

export interface DetectionFilters {
  status?: string
  confidence?: string
  magnitude?: string
  dateRange?: string
  algorithm?: string
  observationId?: string
}

export interface WorkflowFilters {
  status?: string
  type?: string
  dateRange?: string
  userId?: string
}

// Form types
export interface CreateUserRequest {
  email: string
  name: string
  role: string
  permissions: string[]
}

export interface UpdateUserRequest {
  name?: string
  role?: string
  permissions?: string[]
  status?: string
}

export interface UserSettings {
  email: string
  role: string
  notifications: {
    email: boolean
    push: boolean
    weekly: boolean
  }
  preferences: {
    theme: string
    timezone: string
    dateFormat: string
  }
}

// Real-time data types
export interface RealTimeData {
  observations: {
    new: Observation[]
    updated: Observation[]
    deleted: string[]
  }
  detections: {
    new: Detection[]
    updated: Detection[]
    deleted: string[]
  }
  workflows: {
    started: Workflow[]
    completed: Workflow[]
    failed: Workflow[]
  }
  system: {
    alerts: Alert[]
    metrics: SystemMetrics
    status: SystemStatus
  }
}

export interface Alert {
  id: string
  type: 'info' | 'warning' | 'error' | 'success'
  title: string
  message: string
  timestamp: string
  acknowledged: boolean
  source: string
}

export interface SystemStatus {
  status: 'online' | 'degraded' | 'offline'
  services: {
    database: 'up' | 'down'
    api: 'up' | 'down'
    storage: 'up' | 'down'
    ml: 'up' | 'down'
  }
  lastCheck: string
}

// Configuration types
export interface APIClient {
  baseURL: string
  timeout: number
  retries: number
  headers: Record<string, string>
}

export interface AuthConfig {
  supabaseUrl: string
  supabaseKey: string
  redirectTo: string
  cookieName: string
}

export interface StreamingConfig {
  websocket: {
    url: string
    reconnectInterval: number
    maxReconnectAttempts: number
  }
  sse: {
    url: string
    retryInterval: number
  }
  polling: {
    interval: number
    enabled: boolean
  }
}

// Component prop types
export interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  children: React.ReactNode
  onClick?: () => void
  className?: string
}

export interface InputProps {
  type?: 'text' | 'email' | 'password' | 'number' | 'date'
  placeholder?: string
  value: string
  onChange: (value: string) => void
  disabled?: boolean
  error?: string
  className?: string
}

export interface ModalProps {
  isOpen: boolean
  onClose: () => void
  title: string
  children: React.ReactNode
  size?: 'sm' | 'md' | 'lg' | 'xl'
}

export interface TableProps<T> {
  data: T[]
  columns: Column<T>[]
  onRowClick?: (row: T) => void
  loading?: boolean
  emptyMessage?: string
}

export interface Column<T> {
  key: keyof T | string
  title: string
  render?: (value: any, row: T) => React.ReactNode
  sortable?: boolean
  width?: string
}

export interface ChartProps {
  data: any[]
  type: 'line' | 'bar' | 'pie' | 'scatter'
  width?: number
  height?: number
  options?: Record<string, any>
}

export interface CardProps {
  title?: string
  children: React.ReactNode
  className?: string
  actions?: React.ReactNode
}

export interface BadgeProps {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info'
  children: React.ReactNode
  className?: string
}

export interface SpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

// Dashboard specific types
export interface DashboardLayout {
  sidebar: {
    navigation: NavigationItem[]
    userMenu: UserMenuProps
    collapsible: boolean
  }
  header: {
    breadcrumbs: BreadcrumbItem[]
    notifications: NotificationItem[]
    userProfile: UserProfileProps
  }
  main: {
    content: React.ReactNode
    sidebar: React.ReactNode
  }
  footer: {
    status: SystemStatus
    links: FooterLink[]
  }
}

export interface NavigationItem {
  name: string
  href: string
  icon: React.ComponentType<{ className?: string }>
  current?: boolean
  children?: NavigationItem[]
}

export interface BreadcrumbItem {
  name: string
  href?: string
  current?: boolean
}

export interface NotificationItem {
  id: string
  title: string
  message: string
  type: 'info' | 'warning' | 'error' | 'success'
  timestamp: string
  read: boolean
}

export interface UserMenuProps {
  user: User
  onSignOut: () => void
}

export interface UserProfileProps {
  user: User
  onEdit: () => void
}

export interface FooterLink {
  name: string
  href: string
  external?: boolean
}

// API Service interfaces
export interface ObservationService {
  list: (filters: ObservationFilters) => Promise<Observation[]>
  get: (id: string) => Promise<Observation>
  create: (observation: Partial<Observation>) => Promise<Observation>
  update: (id: string, observation: Partial<Observation>) => Promise<Observation>
  delete: (id: string) => Promise<void>
  stats: () => Promise<SystemMetrics['observations']>
  search: (query: string) => Promise<Observation[]>
}

export interface DetectionService {
  list: (filters: DetectionFilters) => Promise<Detection[]>
  get: (id: string) => Promise<Detection>
  create: (detection: Partial<Detection>) => Promise<Detection>
  update: (id: string, detection: Partial<Detection>) => Promise<Detection>
  delete: (id: string) => Promise<void>
  stats: () => Promise<SystemMetrics['detections']>
  visualize: (id: string) => Promise<Detection>
}

export interface WorkflowService {
  list: (filters: WorkflowFilters) => Promise<Workflow[]>
  get: (id: string) => Promise<Workflow>
  create: (workflow: Partial<Workflow>) => Promise<Workflow>
  start: (id: string) => Promise<void>
  stop: (id: string) => Promise<void>
  pause: (id: string) => Promise<void>
  resume: (id: string) => Promise<void>
  logs: (id: string) => Promise<string[]>
}

export interface UserService {
  list: () => Promise<User[]>
  get: (id: string) => Promise<User>
  create: (user: CreateUserRequest) => Promise<User>
  update: (id: string, user: UpdateUserRequest) => Promise<User>
  delete: (id: string) => Promise<void>
  updateSettings: (settings: UserSettings) => Promise<void>
}

export interface SurveyService {
  list: () => Promise<Survey[]>
  get: (id: string) => Promise<Survey>
  sync: (id: string) => Promise<void>
  configure: (id: string, config: Partial<Survey['configuration']>) => Promise<void>
}
