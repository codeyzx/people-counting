/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        status: {
          online: '#10b981',
          offline: '#6b7280',
        },
        crowd: {
          low: '#10b981',
          medium: '#f59e0b',
          high: '#ef4444',
        },
        connection: {
          connecting: '#f59e0b',
          connected: '#10b981',
          disconnected: '#ef4444',
          error: '#ef4444',
        }
      }
    },
  },
  plugins: [],
}
