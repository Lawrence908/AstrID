# AstrID Frontend - Planning Dashboard

A Next.js frontend for displaying AstrID system diagrams, documentation, and planning materials.

## Features

- **System Diagrams**: Interactive SVG diagram viewer for all system architecture diagrams
- **Documentation**: Markdown documentation viewer with syntax highlighting
- **Data Models**: Code viewer for Python models and schemas
- **Responsive Design**: Modern, dark-themed interface optimized for development

## Quick Start

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
cd frontend
npm install
```

### Development

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Production Build

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── app/
│   ├── globals.css          # Global styles and Tailwind config
│   ├── layout.tsx           # Root layout component
│   └── page.tsx             # Main dashboard page
├── components/
│   ├── DiagramViewer.tsx    # SVG diagram display component
│   └── DocumentationViewer.tsx # Markdown/code viewer component
├── public/
│   └── docs/                # Static files (diagrams, docs, models)
└── package.json
```

## Available Content

### Diagrams
- Data Flow Pipeline
- Database Schema
- Domain Interactions
- External Integrations
- System Architecture
- Workflow Orchestration
- Sequence Diagrams
- Linear Tickets Mapping

### Documentation
- Architecture Overview
- Database Schema Design
- Design Overview
- Development Guide
- Linear Tickets
- Logging Guide
- Migration Strategy
- Tech Stack

### Data Models
- Consolidated Models (Python)

## Customization

### Adding New Content

1. Add files to `public/docs/` directory
2. Update the sections array in `app/page.tsx`
3. Restart the development server

### Styling

The app uses Tailwind CSS with a custom AstrID theme:
- Primary Blue: `#2FA4E7`
- Dark Background: `#0D1117`
- Gray Accent: `#BABDBF`

### API Integration

The frontend is configured to proxy API requests to `localhost:8000` (your FastAPI backend) via Next.js rewrites in `next.config.js`.

## Development Notes

- The frontend serves static files from the `public/` directory
- Diagrams are loaded as SVG content and rendered inline
- Documentation uses `react-markdown` with GitHub Flavored Markdown support
- Code files are displayed with syntax highlighting
- All content is loaded dynamically from the file system

## Troubleshooting

### Diagrams Not Loading
- Ensure SVG files are in `public/docs/diagrams/`
- Check browser console for CORS or file loading errors
- Verify file paths in the sections array

### Documentation Not Rendering
- Ensure markdown files are in `public/docs/`
- Check for special characters or encoding issues
- Verify file permissions

### Styling Issues
- Run `npm run build` to ensure Tailwind CSS is properly compiled
- Check for conflicting CSS classes
- Verify Tailwind configuration in `tailwind.config.js`
