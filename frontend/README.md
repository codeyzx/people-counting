# Real-time Device Monitoring Dashboard

Dashboard frontend real-time untuk monitoring status semua device person detection. Dashboard ini menampilkan informasi status device, crowd count, dan crowd status secara real-time menggunakan WebSocket.

## Features

- 📊 **Real-time Monitoring**: Update otomatis via WebSocket tanpa refresh halaman
- 🟢 **Device Status**: Indikator online/offline untuk setiap device
- 👥 **Crowd Count**: Jumlah orang yang terdeteksi saat ini
- 🚦 **Crowd Status**: Kategori kepadatan (Low/Medium/High)
- ⏰ **Last Seen**: Timestamp terakhir kali device mengirim data
- 📱 **Responsive Design**: Bekerja di desktop, tablet, dan mobile
- 🔄 **Auto Reconnection**: Reconnect otomatis dengan exponential backoff

## Technology Stack

- **React 18** + **TypeScript** - Frontend framework
- **Tailwind CSS** - Styling
- **Vite** - Build tool
- **date-fns** - Date formatting
- **Native WebSocket API** - Real-time communication

## Prerequisites

- Node.js 18+ dan npm
- WebSocket server yang mengirim detection events

## Installation

1. Clone repository dan masuk ke folder frontend:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Copy file environment variables:

```bash
cp .env.example .env
```

4. Edit `.env` dan sesuaikan WebSocket URL:

```env
VITE_WS_URL=ws://localhost:8000/ws
```

## Development

Jalankan development server:

```bash
npm run dev
```

Dashboard akan tersedia di `http://localhost:5173`

## Build for Production

Build aplikasi untuk production:

```bash
npm run build
```

File hasil build akan tersedia di folder `dist/`

Preview production build:

```bash
npm run preview
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_WS_URL` | WebSocket server URL | `ws://localhost:8000/ws` |
| `VITE_OFFLINE_THRESHOLD` | Offline detection threshold (ms) | `30000` (30 seconds) |
| `VITE_CHECK_INTERVAL` | Device status check interval (ms) | `5000` (5 seconds) |

## WebSocket Message Format

Dashboard menerima pesan WebSocket dengan format berikut:

```json
{
  "timestamp": "2024-12-02T14:30:00.000Z",
  "source_id": "camera_01",
  "frame_number": 1234,
  "event_type": "update",
  "current_count": 8,
  "tracked_persons": [
    {
      "person_id": 1,
      "bbox": [100, 200, 300, 400],
      "confidence": 0.95,
      "centroid": [200, 300]
    }
  ],
  "metadata": {}
}
```

### Required Fields

- `timestamp`: ISO 8601 formatted timestamp
- `source_id`: Unique device identifier
- `frame_number`: Current frame number
- `event_type`: Event type (`update`, `entry`, `exit`, `lifecycle`)
- `current_count`: Number of people detected
- `tracked_persons`: Array of tracked person objects

## Crowd Status Thresholds

Dashboard mengkategorikan crowd density berdasarkan jumlah orang:

- **Low** (🟢): 0-5 people
- **Medium** (🟡): 6-15 people
- **High** (🔴): >15 people

## Device Status

Device dianggap **offline** jika tidak mengirim data selama lebih dari 30 detik (configurable via `VITE_OFFLINE_THRESHOLD`).

Status dicek secara periodik setiap 5 detik (configurable via `VITE_CHECK_INTERVAL`).

## Project Structure

```
frontend/
├── src/
│   ├── components/          # React components
│   │   ├── ConnectionStatus.tsx
│   │   ├── CrowdStatusBadge.tsx
│   │   ├── DeviceCard.tsx
│   │   ├── DeviceGrid.tsx
│   │   ├── EmptyState.tsx
│   │   └── StatusBadge.tsx
│   ├── hooks/              # Custom React hooks
│   │   ├── useDeviceState.tsx
│   │   ├── useOfflineDetection.ts
│   │   └── useWebSocket.ts
│   ├── types/              # TypeScript type definitions
│   │   └── index.ts
│   ├── utils/              # Utility functions
│   │   ├── crowdStatus.ts
│   │   ├── deviceStatus.ts
│   │   ├── timeFormat.ts
│   │   └── index.ts
│   ├── App.tsx             # Main application component
│   ├── main.tsx            # Application entry point
│   └── index.css           # Global styles
├── public/                 # Static assets
├── .env                    # Environment variables
├── .env.example            # Environment variables template
├── index.html              # HTML template
├── package.json            # Dependencies
├── tailwind.config.js      # Tailwind configuration
├── tsconfig.json           # TypeScript configuration
└── vite.config.ts          # Vite configuration
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

WebSocket API support required.

## Troubleshooting

### Dashboard tidak menerima data

1. Pastikan WebSocket server berjalan
2. Periksa URL di `.env` sudah benar
3. Buka browser console untuk melihat error messages
4. Pastikan tidak ada firewall yang memblokir WebSocket connection

### Device selalu offline

1. Periksa threshold di `.env` (`VITE_OFFLINE_THRESHOLD`)
2. Pastikan device mengirim data secara regular
3. Periksa timestamp format dari device (harus ISO 8601)

### Connection error

1. Pastikan WebSocket server accessible dari browser
2. Jika menggunakan HTTPS, pastikan WebSocket menggunakan WSS (secure)
3. Periksa CORS settings di WebSocket server

## License

MIT

## Support

Untuk pertanyaan atau issues, silakan buat issue di repository.
