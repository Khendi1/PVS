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
