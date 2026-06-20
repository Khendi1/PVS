/*
 * Video Synth — real-time collaborative visual art synthesizer.
 * Copyright (C) 2026 Kyle Henderson
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import { useState, useEffect, useRef } from 'react'
import { wsStreamUrl } from '../api.js'

export default function VideoPreview() {
  const [blobUrl, setBlobUrl] = useState(null)
  const [online, setOnline] = useState(false)
  const wsRef = useRef(null)
  const prevBlobRef = useRef(null)

  useEffect(() => {
    let reconnectTimer = null

    function connect() {
      const ws = new WebSocket(wsStreamUrl())
      wsRef.current = ws

      ws.binaryType = 'blob'

      ws.onopen = () => setOnline(true)

      ws.onmessage = (e) => {
        const url = URL.createObjectURL(e.data)
        setBlobUrl(url)
        if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current)
        prevBlobRef.current = url
      }

      ws.onerror = () => {}

      ws.onclose = () => {
        setOnline(false)
        reconnectTimer = setTimeout(connect, 3000)
      }
    }

    connect()

    return () => {
      clearTimeout(reconnectTimer)
      if (wsRef.current) wsRef.current.close()
      if (prevBlobRef.current) URL.revokeObjectURL(prevBlobRef.current)
    }
  }, [])

  return (
    <div className="video-quadrant">
      <img
        src={blobUrl}
        alt="live output"
        style={{ display: online && blobUrl ? 'block' : 'none' }}
      />
      {(!online || !blobUrl) && (
        <div className="video-offline">
          stream unavailable<br />
          <span style={{ fontSize: 10, marginTop: 4 }}>start synth with --api</span>
        </div>
      )}
    </div>
  )
}
