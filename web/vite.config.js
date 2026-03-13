import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,   // listen on 0.0.0.0 so other machines on the LAN can connect
    proxy: {
      '/params': 'http://127.0.0.1:8000',
      '/snapshot': 'http://127.0.0.1:8000',
      '/stream': 'http://127.0.0.1:8000',
      '/patch': 'http://127.0.0.1:8000',
      '/midi': 'http://127.0.0.1:8000',
      '/lfo': 'http://127.0.0.1:8000',
    }
  }
})
