# AstrID Database Migration Strategy

## Overview

This document outlines the migration strategy from the current basic `ExampleModel` to the comprehensive AstrID database schema. The migration is designed to be incremental and safe, allowing for testing and validation at each step.

## Current State

- **Existing**: `ExampleModel` in `src/adapters/db/models.py`
- **Target**: Comprehensive schema with 12+ tables supporting the full AstrID pipeline
- **Database**: PostgreSQL with SQLAlchemy 2 + Alembic

## Migration Phases

### Phase 1: Core Foundation (Week 1)
**Priority**: Critical - Foundation for all other work

**Tables to Create**:
1. `surveys` - Survey registry
2. `observations` - Core observation data
3. `system_config` - System configuration

**Migration Steps**:
1. Create new models in `models_v2.py`
2. Generate Alembic migration
3. Test migration on development database
4. Update existing code to use new models
5. Remove `ExampleModel` after validation

**Validation**:
- [ ] All existing functionality works
- [ ] New observation ingestion works
- [ ] Basic queries perform well
- [ ] No data loss

### Phase 2: Processing Pipeline (Week 2)
**Priority**: High - Enables preprocessing and differencing

**Tables to Create**:
1. `preprocess_runs` - Preprocessing tracking
2. `difference_runs` - Differencing tracking
3. `candidates` - Potential transients
4. `processing_jobs` - Workflow management

**Migration Steps**:
1. Add new models to `models_v2.py`
2. Generate migration
3. Update preprocessing workflows
4. Update differencing workflows
5. Test end-to-end processing

**Validation**:
- [ ] Preprocessing pipeline works
- [ ] Differencing produces candidates
- [ ] Job tracking functions correctly
- [ ] Performance is acceptable

### Phase 3: ML Integration (Week 3)
**Priority**: High - Enables ML model tracking and inference

**Tables to Create**:
1. `models` - ML model registry
2. `model_runs` - Inference tracking
3. `detections` - ML-identified anomalies

**Migration Steps**:
1. Add ML models to `models_v2.py`
2. Generate migration
3. Update U-Net integration
4. Implement model registry
5. Test ML inference pipeline

**Validation**:
- [ ] Model registry works
- [ ] U-Net inference tracked
- [ ] Detections stored correctly
- [ ] MLflow integration functional

### Phase 4: Validation & Curation (Week 4)
**Priority**: Medium - Enables human validation

**Tables to Create**:
1. `validation_events` - Human review
2. `alerts` - Notifications
3. `audit_log` - System audit trail

**Migration Steps**:
1. Add validation models
2. Generate migration
3. Implement validation workflows
4. Set up alert system
5. Test curation pipeline

**Validation**:
- [ ] Human validation works
- [ ] Alerts are sent correctly
- [ ] Audit trail is complete
- [ ] Curation workflow functions

## Migration Implementation

### 1. Model Transition Strategy

**Option A: Gradual Replacement (Recommended)**
```python
# Keep both models during transition
from .models import ExampleModel  # Old
from .models_v2 import Observation, Survey  # New

# Gradually migrate code to use new models
# Remove old models after full migration
```

**Option B: Direct Replacement**
```python
# Replace models.py entirely with models_v2.py
# Higher risk but cleaner
```

### 2. Alembic Migration Scripts

**Initial Migration**:
```bash
# Generate initial migration
alembic revision --autogenerate -m "Initial comprehensive schema"

# Review generated migration
# Edit if needed for custom constraints

# Apply migration
alembic upgrade head
```

**Data Migration** (if needed):
```python
# Custom data migration script
def upgrade():
    # Migrate any existing data from ExampleModel
    # Set up initial survey data
    # Configure system settings
    pass
```

### 3. Code Updates Required

**Repository Layer**:
```python
# Update repositories to use new models
class ObservationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, observation_data: dict) -> Observation:
        # Use new Observation model
        pass
```

**API Routes**:
```python
# Update API routes to use new models
@router.post("/observations/")
async def create_observation(
    observation: ObservationCreate,
    db: AsyncSession = Depends(get_db)
):
    # Use new repository and models
    pass
```

**Domain Services**:
```python
# Update domain services
class ObservationService:
    def __init__(self, repo: ObservationRepository):
        self.repo = repo
    
    async def process_observation(self, data: dict):
        # Use new models and relationships
        pass
```

## Testing Strategy

### 1. Unit Tests
- [ ] Test all new model relationships
- [ ] Test constraints and validations
- [ ] Test repository methods
- [ ] Test domain services

### 2. Integration Tests
- [ ] Test database migrations
- [ ] Test API endpoints
- [ ] Test workflow pipelines
- [ ] Test ML integration

### 3. Performance Tests
- [ ] Test query performance
- [ ] Test bulk operations
- [ ] Test concurrent access
- [ ] Test memory usage

### 4. End-to-End Tests
- [ ] Test complete observation pipeline
- [ ] Test detection workflow
- [ ] Test validation process
- [ ] Test alert system

## Rollback Strategy

### 1. Database Rollback
```bash
# Rollback to previous migration
alembic downgrade -1

# Or rollback to specific revision
alembic downgrade <revision_id>
```

### 2. Code Rollback
```bash
# Revert to previous commit
git revert <commit_hash>

# Or checkout previous version
git checkout <previous_commit>
```

### 3. Data Backup
```bash
# Backup database before migration
pg_dump astrid_db > backup_before_migration.sql

# Restore if needed
psql astrid_db < backup_before_migration.sql
```

## Risk Mitigation

### 1. Development Environment
- [ ] Test all migrations on development database
- [ ] Validate all functionality before production
- [ ] Performance test with realistic data volumes

### 2. Staging Environment
- [ ] Deploy to staging first
- [ ] Run full test suite
- [ ] Validate with real data
- [ ] Monitor performance

### 3. Production Deployment
- [ ] Schedule maintenance window
- [ ] Backup production database
- [ ] Deploy during low-traffic period
- [ ] Monitor closely after deployment

## Success Criteria

### Phase 1 Success
- [ ] All existing functionality preserved
- [ ] New observation ingestion works
- [ ] Database performance acceptable
- [ ] No data loss

### Phase 2 Success
- [ ] Processing pipeline functional
- [ ] Candidates generated correctly
- [ ] Job tracking works
- [ ] Performance maintained

### Phase 3 Success
- [ ] ML models tracked properly
- [ ] Inference results stored
- [ ] Detections created correctly
- [ ] MLflow integration works

### Phase 4 Success
- [ ] Validation workflow functional
- [ ] Alerts sent correctly
- [ ] Audit trail complete
- [ ] System fully operational

## Timeline

| Phase | Duration | Dependencies | Deliverables |
|-------|----------|--------------|--------------|
| Phase 1 | 1 week | None | Core tables, basic functionality |
| Phase 2 | 1 week | Phase 1 | Processing pipeline |
| Phase 3 | 1 week | Phase 2 | ML integration |
| Phase 4 | 1 week | Phase 3 | Validation & curation |

**Total Duration**: 4 weeks
**Risk Buffer**: 1 week
**Total Timeline**: 5 weeks

## Next Steps

1. **Review and Approve**: Review this migration strategy
2. **Set Up Environment**: Prepare development and staging environments
3. **Begin Phase 1**: Start with core foundation tables
4. **Monitor Progress**: Track progress against success criteria
5. **Iterate**: Adjust strategy based on learnings

## Questions for Discussion

1. **Timeline**: Is 5 weeks acceptable for the migration?
2. **Risk Tolerance**: Are we comfortable with the rollback strategy?
3. **Testing**: Do we need additional testing environments?
4. **Performance**: What are the performance requirements?
5. **Data Volume**: What's the expected data volume for testing?
