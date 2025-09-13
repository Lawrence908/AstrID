# AstrID Design Diagrams

This directory contains PlantUML diagrams that document the AstrID system architecture, data flows, and implementation details.

## Diagram Overview

### 1. System Architecture (`system-architecture.puml`)
**Purpose**: High-level view of the entire AstrID system
**Shows**:
- Frontend layer (Next.js dashboard)
- API Gateway layer (FastAPI)
- Domain services layer (6 bounded contexts)
- Infrastructure layer (databases, storage, message queues)
- External service integrations

**Key Insights**:
- Clear separation of concerns across layers
- Domain-driven design with bounded contexts
- Event-driven architecture with Redis/Dramatiq
- External API integrations for astronomical data

### 2. Data Flow Pipeline (`data-flow-pipeline.puml`)
**Purpose**: End-to-end data processing flow
**Shows**:
- Complete pipeline from observation ingestion to cataloging
- Linear progression through all processing stages
- Integration points with external services
- Storage and database operations

**Key Insights**:
- Sequential processing pipeline
- Clear stage boundaries
- Error handling and retry mechanisms
- Human validation integration

### 3. Domain Interactions (`domain-interactions.puml`)
**Shows**:
- How domains interact with each other
- Entity relationships and dependencies
- Event-driven communication patterns
- Infrastructure integration points

**Key Insights**:
- Loose coupling between domains
- Event-based communication
- Shared infrastructure services
- Clear data flow between domains

### 4. External Integrations (`external-integrations.puml`)
**Purpose**: External service integration details
**Shows**:
- MAST API integration for observations
- SkyView API for reference images
- Cloudflare R2 storage integration
- Data flow between external services and internal components

**Key Insights**:
- Clean abstraction of external services
- Content-addressable storage pattern
- Multi-wavelength data handling
- Reference image management

### 5. Database Schema (`database-schema.puml`)
**Purpose**: Complete database design
**Shows**:
- All 13 database models across 6 domains
- Entity relationships and foreign keys
- Data types and constraints
- Indexing strategy

**Key Insights**:
- Domain-driven database design
- UUID primary keys throughout
- JSONB for flexible metadata
- Spatial indexing for astronomical coordinates

### 6. Workflow Orchestration (`workflow-orchestration.puml`)
**Purpose**: Background processing and workflow management
**Shows**:
- Prefect flows for complex workflows
- Dramatiq workers for background tasks
- Redis event system
- Error handling and retry policies

**Key Insights**:
- Event-driven workflow orchestration
- Robust error handling
- Scalable worker architecture
- Monitoring and alerting integration

### 7. Sequence Diagram (`sequence-observation-processing.puml`)
**Purpose**: Detailed interaction sequence for observation processing
**Shows**:
- Step-by-step interaction between components
- Data flow through the system
- Error handling and status updates
- Human validation integration

**Key Insights**:
- Clear component interactions
- Asynchronous processing patterns
- Status tracking throughout pipeline
- Human-in-the-loop validation

### 8. Linear Tickets Mapping (`linear-tickets-mapping.puml`)
**Purpose**: Maps Linear tickets to architecture components
**Shows**:
- Phase-based development approach
- Ticket dependencies and relationships
- Architecture component mapping
- Development priority ordering

**Key Insights**:
- Phased development approach
- Clear dependency management
- Architecture-driven ticket organization
- Risk mitigation through proper sequencing

## How to Use These Diagrams

### For Development Planning
1. **Start with System Architecture** - Understand the big picture
2. **Review Data Flow Pipeline** - Understand the processing flow
3. **Check Domain Interactions** - Understand component relationships
4. **Reference Database Schema** - Understand data structure

### For Implementation
1. **Use External Integrations** - Understand API requirements
2. **Follow Sequence Diagram** - Implement step-by-step processing
3. **Reference Workflow Orchestration** - Set up background processing
4. **Check Linear Tickets Mapping** - Follow development phases

### For Architecture Decisions
1. **Domain Interactions** - Ensure proper separation of concerns
2. **Database Schema** - Maintain data integrity
3. **Workflow Orchestration** - Ensure scalable processing
4. **System Architecture** - Maintain architectural principles

## Diagram Maintenance

### When to Update
- **System Architecture**: When adding new domains or major components
- **Data Flow Pipeline**: When changing processing steps
- **Domain Interactions**: When changing domain boundaries
- **Database Schema**: When adding/modifying models
- **External Integrations**: When adding new external services
- **Workflow Orchestration**: When changing processing patterns
- **Sequence Diagrams**: When changing component interactions
- **Linear Tickets Mapping**: When adding new tickets or changing phases

### How to Update
1. Edit the `.puml` files using PlantUML syntax
2. Test diagrams using PlantUML tools or VS Code extensions
3. Update this README if adding new diagrams
4. Commit changes with descriptive messages

## Tools for Viewing

### VS Code Extensions
- **PlantUML**: Renders diagrams in VS Code
- **PlantUML Previewer**: Live preview of diagrams

### Online Tools
- **PlantUML Online Server**: https://www.plantuml.com/plantuml/uml/
- **PlantText**: https://www.planttext.com/

### Command Line
```bash
# Install PlantUML
npm install -g plantuml

# Generate PNG from PUML
plantuml docs/diagrams/*.puml

# Generate SVG from PUML
plantuml -tsvg docs/diagrams/*.puml
```

## Integration with Development

These diagrams should be:
1. **Referenced during code reviews** - Ensure implementation matches design
2. **Updated during refactoring** - Keep diagrams current
3. **Used for onboarding** - Help new developers understand the system
4. **Referenced in documentation** - Link to relevant diagrams
5. **Used for planning** - Guide architectural decisions

## Next Steps

As you develop the system:
1. **Keep diagrams current** - Update as you implement
2. **Add detail diagrams** - Create specific diagrams for complex areas
3. **Document decisions** - Add notes explaining architectural choices
4. **Share with team** - Use diagrams for communication and planning
5. **Iterate and improve** - Refine diagrams based on implementation experience

