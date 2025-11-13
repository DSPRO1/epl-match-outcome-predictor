# EPL Match Predictor - Web UI

Beautiful web interface for the EPL Match Outcome Predictor API, built with Astro and React.

## Features

- Clean, modern UI with Tailwind CSS
- Real-time match outcome predictions
- Simple mode for quick predictions (just team names)
- Advanced mode with ELO ratings, form stats, and rest days
- Interactive probability visualization
- Responsive design

## Setup

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API key:
   ```
   PUBLIC_API_URL=https://dspro1--epl-predictor-fastapi-app.modal.run
   PUBLIC_API_KEY=your-api-key-here
   ```

3. **Start development server:**
   ```bash
   npm run dev
   ```

   Visit http://localhost:4321

## Build for Production

```bash
npm run build
npm run preview  # Preview production build locally
```

## Deployment

This Astro site can be deployed to:
- **Vercel**: `vercel deploy`
- **Netlify**: `netlify deploy`
- **Cloudflare Pages**: Connect via Git
- **GitHub Pages**: See [Astro docs](https://docs.astro.build/en/guides/deploy/)

Make sure to set the environment variables in your deployment platform:
- `PUBLIC_API_URL`
- `PUBLIC_API_KEY`

## Tech Stack

- **Astro 5.x** - Static site generator
- **React 18** - Interactive components
- **Tailwind CSS 4** - Styling
- **TypeScript** - Type safety

## Project Structure

```
web-ui/
├── src/
│   ├── components/
│   │   └── PredictionForm.tsx    # Main prediction form
│   ├── layouts/
│   │   └── Layout.astro          # Base layout
│   ├── pages/
│   │   └── index.astro           # Home page
│   └── styles/
│       └── global.css            # Global styles
├── public/
│   └── favicon.svg
└── astro.config.mjs              # Astro configuration
```

## API Integration

The UI connects to the Modal.com deployed API endpoint. See `src/components/PredictionForm.tsx` for implementation details.

## License

MIT
