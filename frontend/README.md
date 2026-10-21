# frontend/

Next.js UI for the live show. Today it's the default `create-next-app` page; it grows into:

- **Waveform avatar**: an original animated line driven by the TTS audio amplitude.
- **Operator panel**: session context (`venue`, `energy`, `age`), a manual "laughed" button
  (fallback reward + labels), start/stop, current persona mode and laugh meter.
- **Training mode**: end-of-episode rating and pairwise "which bit was better?" votes.

## Setup and run

```sh
cd frontend
npm install
npm run dev        # http://localhost:3000
npm run build      # production build check
```

The backend must be running on `http://localhost:8000` for the websocket (CORS allows
`http://localhost:3000`).

## Stack

Next.js 15.5 (App Router, TypeScript, `src/` dir, ESLint, no Tailwind). Pinned to a patched
15.5.x release (CVE-2025-66478); upgrade within 15.5 with `npm install next@15.5 eslint-config-next@15.5`.

## Planned structure

| Path | Role |
|---|---|
| `src/app/page.tsx` | Show view: waveform + operator panel |
| `src/components/Waveform.tsx` | Canvas line animated from audio amplitude |
| `src/components/OperatorPanel.tsx` | Context dropdowns, laugh button, status |
| `src/lib/ws.ts` | Websocket client to the backend |
