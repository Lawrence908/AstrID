'use client'

import { useState } from 'react'
import Link from 'next/link'
import { ArrowLeft, GitBranch, ExternalLink, Calendar, User, Clock, CheckCircle, Circle, AlertCircle, Filter, Search, FileText, Globe, List } from 'lucide-react'
import DocumentationViewer from '@/components/DocumentationViewer'

const tickets = [
  // Foundation & Infrastructure Setup
  {
    id: 'ASTR-69',
    title: 'Development Environment Setup',
    description: 'Set up complete development environment for AstrID project',
    status: 'completed',
    priority: 'P1',
    assignee: 'Chris Lawrence',
    created: '2025-09-01',
    updated: '2025-09-03',
    labels: ['infrastructure', 'high-priority'],
    progress: 100,
    project: 'ASTRID-INFRA'
  },
  {
    id: 'ASTR-70',
    title: 'Database Setup and Migrations',
    description: 'Design and implement database schema with migration system',
    status: 'completed',
    priority: 'P1',
    assignee: 'Chris Lawrence',
    created: '2025-09-02',
    updated: '2025-09-05',
    labels: ['infrastructure', 'database', 'high-priority'],
    progress: 100,
    project: 'ASTRID-INFRA'
  },
  {
    id: 'ASTR-71',
    title: 'Cloud Storage Integration',
    description: 'Configure cloud storage for datasets and artifacts',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-03',
    updated: '2025-09-05',
    labels: ['infrastructure'],
    progress: 100,
    project: 'ASTRID-INFRA'
  },
  {
    id: 'ASTR-72',
    title: 'Supabase Integration',
    description: 'Implement authentication and authorization system',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-04',
    updated: '2025-09-06',
    labels: ['infrastructure', 'security'],
    progress: 100,
    project: 'ASTRID-INFRA'
  },

  // Core Domain Implementation
  {
    id: 'ASTR-73',
    title: 'Observation Models and Services',
    description: 'Implement core observation domain models and business logic',
    status: 'completed',
    priority: 'P1',
    assignee: 'Chris Lawrence',
    created: '2025-09-06',
    updated: '2025-09-16',
    labels: ['core-domain', 'high-priority'],
    progress: 100,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-74',
    title: 'Survey Integration',
    description: 'Integrate with external astronomical survey APIs',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-07',
    updated: '2025-09-16',
    labels: ['core-domain', 'integration'],
    progress: 100,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-75',
    title: 'FITS Processing Pipeline',
    description: 'Implement FITS file processing and WCS handling',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-08',
    updated: '2025-09-16',
    labels: ['core-domain', 'data-processing'],
    progress: 100,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-76',
    title: 'Image Preprocessing Services',
    description: 'Implement image calibration and preprocessing pipeline',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-10',
    updated: '2025-09-16',
    labels: ['core-domain', 'image-processing'],
    progress: 100,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-77',
    title: 'Astronomical Image Processing',
    description: 'Advanced image processing with OpenCV and scikit-image',
    status: 'completed',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-11',
    updated: '2025-09-16',
    labels: ['core-domain', 'image-processing'],
    progress: 100,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-78',
    title: 'Image Differencing Algorithms',
    description: 'Implement image differencing algorithms for anomaly detection',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-12',
    updated: '2025-09-16',
    labels: ['core-domain', 'algorithm'],
    progress: 100,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-79',
    title: 'Source Extraction',
    description: 'Extract and analyze sources from difference images',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-13',
    updated: '2025-09-17',
    labels: ['core-domain', 'algorithm'],
    progress: 100,
    project: 'ASTRID-CORE'
  },

  // Detection Domain
  {
    id: 'ASTR-80',
    title: 'U-Net Model Integration',
    description: 'Integrate existing U-Net model into new architecture',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-14',
    updated: '2025-09-17',
    labels: ['ml', 'model-integration'],
    progress: 100,
    project: 'ASTRID-ML'
  },
  {
    id: 'ASTR-81',
    title: 'Anomaly Detection Pipeline',
    description: 'Complete anomaly detection service implementation',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-15',
    updated: '2025-09-17',
    labels: ['ml', 'pipeline'],
    progress: 100,
    project: 'ASTRID-ML'
  },

  // Curation Domain
  {
    id: 'ASTR-82',
    title: 'Human Validation System',
    description: 'Create human validation interface for detected anomalies',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-18',
    updated: '2025-09-18',
    labels: ['core-domain', 'ui'],
    progress: 0,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-83',
    title: 'Data Cataloging',
    description: 'Implement data cataloging and export functionality',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-19',
    updated: '2025-09-19',
    labels: ['core-domain', 'data'],
    progress: 0,
    project: 'ASTRID-CORE'
  },

  // API & Web Interface
  {
    id: 'ASTR-84',
    title: 'Core API Endpoints',
    description: 'Implement core API endpoints for all domains',
    status: 'completed',
    priority: 'P1',
    assignee: 'Chris Lawrence',
    created: '2025-09-16',
    updated: '2025-09-17',
    labels: ['api', 'high-priority'],
    progress: 100,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-85',
    title: 'API Documentation and Testing',
    description: 'Comprehensive API documentation and testing',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-17',
    updated: '2025-09-17',
    labels: ['api', 'testing'],
    progress: 100,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-86',
    title: 'Next.js Dashboard Setup',
    description: 'Set up Next.js dashboard with authentication',
    status: 'completed',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-17',
    updated: '2025-09-17',
    labels: ['ui', 'frontend'],
    progress: 100,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-87',
    title: 'Dashboard Features',
    description: 'Implement core dashboard functionality',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-18',
    updated: '2025-09-18',
    labels: ['ui', 'frontend'],
    progress: 0,
    project: 'ASTRID-API'
  },

  // Machine Learning & Model Management
  {
    id: 'ASTR-88',
    title: 'MLflow Integration',
    description: 'Set up MLflow for experiment tracking and model management',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-15',
    updated: '2025-09-17',
    labels: ['ml', 'infrastructure'],
    progress: 100,
    project: 'ASTRID-ML'
  },
  {
    id: 'ASTR-89',
    title: 'Model Training Pipeline',
    description: 'Automated model training and optimization workflows',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-18',
    updated: '2025-09-18',
    labels: ['ml', 'pipeline'],
    progress: 0,
    project: 'ASTRID-ML'
  },
  {
    id: 'ASTR-90',
    title: 'Model Serving',
    description: 'Production model serving and monitoring',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-19',
    updated: '2025-09-19',
    labels: ['ml', 'mlops'],
    progress: 0,
    project: 'ASTRID-ML'
  },

  // Workflow & Orchestration
  {
    id: 'ASTR-91',
    title: 'Workflow Orchestration',
    description: 'Set up Prefect for workflow orchestration',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-15',
    updated: '2025-09-17',
    labels: ['workflow', 'orchestration'],
    progress: 100,
    project: 'ASTRID-WORK'
  },
  {
    id: 'ASTR-92',
    title: 'Dramatiq Workers',
    description: 'Implement background processing workers',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-16',
    updated: '2025-09-17',
    labels: ['workflow', 'background'],
    progress: 100,
    project: 'ASTRID-WORK'
  },

  // Testing & Quality Assurance
  {
    id: 'ASTR-93',
    title: 'Test Framework Setup',
    description: 'Set up comprehensive testing infrastructure',
    status: 'completed',
    priority: 'P1',
    assignee: 'Chris Lawrence',
    created: '2025-09-17',
    updated: '2025-09-17',
    labels: ['testing', 'high-priority'],
    progress: 100,
    project: 'ASTRID-TEST'
  },
  {
    id: 'ASTR-94',
    title: 'Test Implementation',
    description: 'Implement comprehensive test coverage',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-18',
    updated: '2025-09-18',
    labels: ['testing'],
    progress: 0,
    project: 'ASTRID-TEST'
  },
  {
    id: 'ASTR-95',
    title: 'Code Quality Tools',
    description: 'Configure code quality and formatting tools',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-17',
    updated: '2025-09-17',
    labels: ['testing', 'quality'],
    progress: 100,
    project: 'ASTRID-TEST'
  },

  // Deployment & Operations
  {
    id: 'ASTR-96',
    title: 'Docker Setup',
    description: 'Containerize all services for deployment',
    status: 'completed',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-17',
    updated: '2025-09-17',
    labels: ['deployment', 'docker'],
    progress: 100,
    project: 'ASTRID-DEPLOY'
  },
  {
    id: 'ASTR-97',
    title: 'Production Setup',
    description: 'Configure production environment and monitoring',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-18',
    updated: '2025-09-18',
    labels: ['deployment', 'production'],
    progress: 0,
    project: 'ASTRID-DEPLOY'
  },
  {
    id: 'ASTR-98',
    title: 'GitHub Actions',
    description: 'Set up automated CI/CD pipeline',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-19',
    updated: '2025-09-19',
    labels: ['deployment', 'ci-cd'],
    progress: 0,
    project: 'ASTRID-DEPLOY'
  },

  // Documentation & Training
  {
    id: 'ASTR-99',
    title: 'Technical Documentation',
    description: 'Create comprehensive technical documentation',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-20',
    updated: '2025-09-20',
    labels: ['documentation'],
    progress: 0,
    project: 'ASTRID-DOCS'
  },
  {
    id: 'ASTR-100',
    title: 'User Training',
    description: 'Create user training materials and onboarding',
    status: 'todo',
    priority: 'P4',
    assignee: 'Chris Lawrence',
    created: '2025-09-21',
    updated: '2025-09-21',
    labels: ['documentation', 'training'],
    progress: 0,
    project: 'ASTRID-DOCS'
  },

  // Additional Features
  {
    id: 'ASTR-101',
    title: 'GPU Energy Tracking for ML Workloads',
    description: 'Implement GPU energy tracking for ML workloads',
    status: 'completed',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-17',
    updated: '2025-09-17',
    labels: ['mlops', 'monitoring', 'improvement'],
    progress: 100,
    project: 'ASTRID-WORK'
  },
  {
    id: 'ASTR-102',
    title: 'Refine Model Performance Tracking',
    description: 'Iterate through ModelRun session and ensure all performance metrics are populated',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-18',
    updated: '2025-09-18',
    labels: ['ml', 'monitoring', 'metrics'],
    progress: 0,
    project: 'ASTRID-ML'
  },
  {
    id: 'ASTR-103',
    title: 'Supabase Connection Pooling Hardening',
    description: 'Minimize per-service connection pools and validate Transaction pooling compatibility',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-19',
    updated: '2025-09-19',
    labels: ['infrastructure', 'database', 'stability'],
    progress: 0,
    project: 'ASTRID-INFRA'
  },
  {
    id: 'ASTR-104',
    title: 'Workers Operations Page & Dashboard Integration',
    description: 'Create dedicated Workers page for operations and monitoring',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-20',
    updated: '2025-09-20',
    labels: ['ui', 'frontend', 'workflow', 'ops'],
    progress: 0,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-105',
    title: 'Testing & Diagnostics Page',
    description: 'Add frontend page to run smoke checks and surface system health',
    status: 'todo',
    priority: 'P3',
    assignee: 'Chris Lawrence',
    created: '2025-09-21',
    updated: '2025-09-21',
    labels: ['ui', 'frontend', 'testing', 'ops'],
    progress: 0,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-106',
    title: 'Training Notebook for Model Training and MLflow Logging',
    description: 'Create Jupyter notebook for initial model training with MLflow integration',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-22',
    updated: '2025-09-22',
    labels: ['ml', 'notebook', 'training'],
    progress: 0,
    project: 'ASTRID-ML'
  },
  {
    id: 'ASTR-107',
    title: 'Task Scheduler for Automated Inference Runs',
    description: 'Set up task scheduler for automated inference pipeline execution',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-23',
    updated: '2025-09-23',
    labels: ['workflow', 'scheduler', 'automation'],
    progress: 0,
    project: 'ASTRID-WORK'
  },
  {
    id: 'ASTR-108',
    title: 'Email Notification for Detected Anomalies',
    description: 'Implement SendGrid email notification system for anomaly detection alerts',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-24',
    updated: '2025-09-24',
    labels: ['api', 'notifications', 'email'],
    progress: 0,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-109',
    title: 'Anomaly Timeline Feature for React Frontend',
    description: 'Add interactive timeline view to display detected anomalies over time',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-25',
    updated: '2025-09-25',
    labels: ['ui', 'frontend', 'timeline', 'visualization'],
    progress: 0,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-110',
    title: 'Virtual Sky Observatory 3D Visualization (Stretch Goal)',
    description: 'Create immersive 3D virtual sky observatory for exploring astronomical data',
    status: 'todo',
    priority: 'P4',
    assignee: 'Chris Lawrence',
    created: '2025-09-26',
    updated: '2025-09-26',
    labels: ['ui', 'frontend', '3d', 'visualization', 'stretch-goal'],
    progress: 0,
    project: 'ASTRID-API'
  },
  {
    id: 'ASTR-111',
    title: 'Expert Review Process for Anomaly Confirmation',
    description: 'Implement expert review workflow for anomaly validation and scientific importance rating',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-27',
    updated: '2025-09-27',
    labels: ['core-domain', 'curation', 'expert-review'],
    progress: 0,
    project: 'ASTRID-CORE'
  },
  {
    id: 'ASTR-112',
    title: 'Data Backup and Disaster Recovery System',
    description: 'Implement comprehensive backup and disaster recovery system for data protection',
    status: 'todo',
    priority: 'P4',
    assignee: 'Chris Lawrence',
    created: '2025-09-28',
    updated: '2025-09-28',
    labels: ['infrastructure', 'backup', 'disaster-recovery'],
    progress: 0,
    project: 'ASTRID-INFRA'
  },
  {
    id: 'ASTR-113',
    title: 'Real Data Loading Integration for Training Pipeline',
    description: 'Implement real data loading functions that integrate with the complete AstrID workflow pipeline',
    status: 'todo',
    priority: 'P2',
    assignee: 'Chris Lawrence',
    created: '2025-09-29',
    updated: '2025-09-29',
    labels: ['ml', 'data-pipeline', 'training'],
    progress: 0,
    project: 'ASTRID-ML'
  }
]

const statuses = ['All', 'todo', 'in-progress', 'completed']
const priorities = ['All', 'P1', 'P2', 'P3', 'P4']
const projects = ['All', 'ASTRID-INFRA', 'ASTRID-CORE', 'ASTRID-API', 'ASTRID-ML', 'ASTRID-WORK', 'ASTRID-TEST', 'ASTRID-DEPLOY', 'ASTRID-DOCS']
const labels = ['All', 'infrastructure', 'high-priority', 'core-domain', 'image-processing', 'preprocessing', 'ml', 'deep-learning', 'detection', 'pipeline', 'mlflow', 'experiment-tracking', 'workflow', 'orchestration', 'dramatiq', 'database', 'integration', 'data-processing', 'algorithm', 'model-integration', 'ui', 'frontend', 'api', 'testing', 'quality', 'deployment', 'docker', 'production', 'ci-cd', 'documentation', 'training', 'mlops', 'monitoring', 'improvement', 'metrics', 'stability', 'ops', 'notebook', 'scheduler', 'automation', 'notifications', 'email', 'timeline', 'visualization', '3d', 'stretch-goal', 'curation', 'expert-review', 'backup', 'disaster-recovery', 'data-pipeline', 'security']

export default function TicketsPage() {
  const [selectedTicket, setSelectedTicket] = useState(tickets[0])
  const [statusFilter, setStatusFilter] = useState('All')
  const [priorityFilter, setPriorityFilter] = useState('All')
  const [projectFilter, setProjectFilter] = useState('All')
  const [labelFilter, setLabelFilter] = useState('All')
  const [searchQuery, setSearchQuery] = useState('')
  const [showFullList, setShowFullList] = useState(false)
  const [view, setView] = useState<'linear' | 'local'>('linear')

  const filteredTickets = tickets.filter(ticket => {
    const matchesStatus = statusFilter === 'All' || ticket.status === statusFilter
    const matchesPriority = priorityFilter === 'All' || ticket.priority === priorityFilter
    const matchesProject = projectFilter === 'All' || ticket.project === projectFilter
    const matchesLabel = labelFilter === 'All' || ticket.labels.includes(labelFilter)
    const matchesSearch = ticket.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         ticket.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         ticket.id.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesStatus && matchesPriority && matchesProject && matchesLabel && matchesSearch
  })

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'in-progress':
        return <Clock className="w-4 h-4 text-yellow-500" />
      case 'todo':
        return <Circle className="w-4 h-4 text-gray-500" />
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-900 text-green-300'
      case 'in-progress':
        return 'bg-yellow-900 text-yellow-300'
      case 'todo':
        return 'bg-gray-700 text-gray-300'
      default:
        return 'bg-gray-700 text-gray-300'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'P1':
        return 'bg-red-900 text-red-300'
      case 'P2':
        return 'bg-yellow-900 text-yellow-300'
      case 'P3':
        return 'bg-green-900 text-green-300'
      default:
        return 'bg-gray-700 text-gray-300'
    }
  }

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-astrid-dark">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center space-x-4 mb-4">
            <Link
              href="/planning"
              className="inline-flex items-center px-3 py-2 text-sm font-medium text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Planning
            </Link>
          </div>
          <div className="flex items-center space-x-6 mb-4">
            <div className="flex-shrink-0">
              <div className="w-16 h-16 bg-astrid-blue/20 rounded-lg flex items-center justify-center">
                <GitBranch className="w-8 h-8 text-astrid-blue" />
              </div>
            </div>
            <div>
              <h1 className="text-4xl font-bold text-white mb-2">Linear Tickets</h1>
              <p className="text-xl text-gray-400">
                Project tickets and progress tracking
              </p>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <a
                href="https://linear.app/astrid-astro-ident/team/ASTR/all"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-3 py-2 text-sm font-medium text-astrid-blue hover:text-blue-400 transition-colors"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                Open in Linear
              </a>
              <div className="bg-gray-800 border border-gray-700 rounded-lg p-1 flex">
                <button
                  onClick={() => setView('linear')}
                  className={`inline-flex items-center px-3 py-1 text-sm rounded-md ${view === 'linear' ? 'bg-astrid-blue text-white' : 'text-gray-300 hover:text-white'}`}
                  title="Linear View"
                >
                  <Globe className="w-4 h-4 mr-2" /> Linear
                </button>
                <button
                  onClick={() => setView('local')}
                  className={`inline-flex items-center px-3 py-1 text-sm rounded-md ${view === 'local' ? 'bg-astrid-blue text-white' : 'text-gray-300 hover:text-white'}`}
                  title="Local View"
                >
                  <List className="w-4 h-4 mr-2" /> Local
                </button>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              <span className="text-sm text-gray-300">{tickets.length} tickets total</span>
            </div>
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Last updated: Sep 29, 2025</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-sm text-gray-300">
                {tickets.filter(t => t.status === 'completed').length} completed ({Math.round((tickets.filter(t => t.status === 'completed').length / tickets.length) * 100)}%)
              </span>
            </div>
          </div>
        </div>

        {view === 'linear' ? (
          <div className="rounded-lg border border-gray-700 overflow-hidden">
            <div className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2 text-gray-300 text-sm">
                <Globe className="w-4 h-4" />
                <span>Embedded Linear workspace</span>
              </div>
              <a
                href="https://linear.app/astrid-astro-ident/team/ASTR/all"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center px-3 py-1 text-xs font-medium text-astrid-blue hover:text-blue-400 transition-colors"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                Open in new tab
              </a>
            </div>
            <div className="bg-gray-900">
              <iframe
                src="https://linear.app/astrid-astro-ident/team/ASTR/all"
                className="w-full h-[calc(100vh-14rem)] bg-white"
                title="Linear Tickets"
                referrerPolicy="no-referrer"
              />
            </div>
            <div className="bg-gray-800 border-t border-gray-700 px-4 py-2 text-xs text-gray-400">
              If the view does not load, Linear may block embedding. Use the
              <a href="https://linear.app/astrid-astro-ident/team/ASTR/all" target="_blank" rel="noopener noreferrer" className="text-astrid-blue hover:text-blue-400 ml-1">direct link</a>.
            </div>
          </div>
        ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Sidebar - Ticket List */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold text-white">Tickets</h2>
                  <button
                    onClick={() => setShowFullList(true)}
                    className="inline-flex items-center px-3 py-2 text-sm font-medium text-astrid-blue hover:text-blue-400 transition-colors"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    View Full List
                  </button>
                </div>

                {/* Search */}
                <div className="relative mb-4">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search tickets..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-astrid-blue focus:border-transparent"
                  />
                </div>

                {/* Filters */}
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Status</label>
                    <div className="flex flex-wrap gap-2">
                      {statuses.map((status) => (
                        <button
                          key={status}
                          onClick={() => setStatusFilter(status)}
                          className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                            statusFilter === status
                              ? 'bg-astrid-blue text-white'
                              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                          }`}
                        >
                          {status === 'All' ? 'All' : status.replace('-', ' ')}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Priority</label>
                    <div className="flex flex-wrap gap-2">
                      {priorities.map((priority) => (
                        <button
                          key={priority}
                          onClick={() => setPriorityFilter(priority)}
                          className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                            priorityFilter === priority
                              ? 'bg-astrid-blue text-white'
                              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                          }`}
                        >
                          {priority === 'All' ? 'All' : priority}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Project</label>
                    <div className="flex flex-wrap gap-2">
                      {projects.map((project) => (
                        <button
                          key={project}
                          onClick={() => setProjectFilter(project)}
                          className={`px-3 py-1 text-xs font-medium rounded-full transition-colors ${
                            projectFilter === project
                              ? 'bg-astrid-blue text-white'
                              : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                          }`}
                        >
                          {project === 'All' ? 'All' : project.replace('ASTRID-', '')}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="p-4 space-y-2 max-h-96 overflow-y-auto">
                {filteredTickets.map((ticket) => (
                  <button
                    key={ticket.id}
                    onClick={() => setSelectedTicket(ticket)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedTicket.id === ticket.id
                        ? 'bg-astrid-blue/20 border border-astrid-blue/50'
                        : 'hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-start space-x-3">
                      <div className="flex-shrink-0 mt-1">
                        {getStatusIcon(ticket.status)}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-1">
                          <h3 className="text-sm font-medium text-white truncate">
                            {ticket.id}
                          </h3>
                          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getPriorityColor(ticket.priority)}`}>
                            {ticket.priority}
                          </span>
                        </div>
                        <p className="text-xs text-gray-300 truncate">
                          {ticket.title}
                        </p>
                        <div className="flex items-center space-x-2 mt-2">
                          <span className="text-xs text-gray-500">{ticket.assignee}</span>
                          <span className="text-xs text-gray-500">•</span>
                          <span className="text-xs text-gray-500">{ticket.progress}%</span>
                        </div>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main Content - Ticket Details */}
          <div className="lg:col-span-2">
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="p-6 border-b border-gray-700">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <h2 className="text-xl font-bold text-white">
                        {selectedTicket.id}
                      </h2>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(selectedTicket.status)}`}>
                        {selectedTicket.status.replace('-', ' ')}
                      </span>
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${getPriorityColor(selectedTicket.priority)}`}>
                        {selectedTicket.priority} priority
                      </span>
                    </div>
                    <h3 className="text-lg font-semibold text-white mb-2">
                      {selectedTicket.title}
                    </h3>
                    <p className="text-gray-400">{selectedTicket.description}</p>
                  </div>
                  <a
                    href={`/docs/tickets/linear-tickets.md#${selectedTicket.id.toLowerCase()}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center px-3 py-2 text-sm font-medium text-astrid-blue hover:text-blue-400 transition-colors"
                  >
                    <ExternalLink className="w-4 h-4 mr-2" />
                    View Details
                  </a>
                </div>
              </div>

              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Assignee</h4>
                    <div className="flex items-center space-x-2">
                      <User className="w-4 h-4 text-gray-400" />
                      <span className="text-white">{selectedTicket.assignee}</span>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Project</h4>
                    <div className="flex items-center space-x-2">
                      <GitBranch className="w-4 h-4 text-gray-400" />
                      <span className="text-white">{selectedTicket.project}</span>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Progress</h4>
                    <div className="flex items-center space-x-2">
                      <div className="flex-1 bg-gray-700 rounded-full h-2">
                        <div
                          className="bg-astrid-blue h-2 rounded-full transition-all duration-300"
                          style={{ width: `${selectedTicket.progress}%` }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-300">{selectedTicket.progress}%</span>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Created</h4>
                    <div className="flex items-center space-x-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span className="text-white">{selectedTicket.created}</span>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-300 mb-2">Last Updated</h4>
                    <div className="flex items-center space-x-2">
                      <Clock className="w-4 h-4 text-gray-400" />
                      <span className="text-white">{selectedTicket.updated}</span>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Labels</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedTicket.labels.map((label) => (
                      <span
                        key={label}
                        className="px-2 py-1 text-xs font-medium bg-gray-700 text-gray-300 rounded-full"
                      >
                        {label}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        )}
      </div>

      {/* Full List Modal */}
      {showFullList && (
        <div className="fixed inset-0 bg-black bg-opacity-90 z-50 overflow-y-auto">
          <div className="min-h-full flex flex-col">
            <div className="flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
              <h2 className="text-xl font-bold text-white">Linear Tickets - Full Documentation</h2>
              <button
                onClick={() => setShowFullList(false)}
                className="inline-flex items-center px-4 py-2 text-sm font-medium text-gray-300 hover:text-white transition-colors"
              >
                <FileText className="w-4 h-4 mr-2" />
                Close
              </button>
            </div>
            <div className="flex-1 p-4">
              <div className="bg-gray-900 rounded-lg p-6 max-w-6xl mx-auto">
                <DocumentationViewer file="/docs/tickets/linear-tickets.md" />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
