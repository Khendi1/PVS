import { useState, useEffect, useRef } from 'react'
import { streamUrl } from '../api.js'

export default function VideoPreview() {
  const [src, setSrc] = useState(streamUrl())
  const [online, setOnline] = useState(true)
  const retryRef = useRef(null)

  function handleError() {
    setOnline(false)
    retryRef.current = setTimeout(() => {
      setSrc(streamUrl())
      setOnline(true)
    }, 3000)
  }

  useEffect(() => () => clearTimeout(retryRef.current), [])

  return (
    <div className="video-quadrant">
      <img
        src={src}
        alt="live output"
        style={{ display: online ? 'block' : 'none' }}
        onError={handleError}
        onLoad={() => setOnline(true)}
      />
      {!online && (
        <div className="video-offline">
          stream unavailable<br />
          <span style={{ fontSize: 10, marginTop: 4 }}>start synth with --api</span>
        </div>
      )}
    </div>
  )
}
