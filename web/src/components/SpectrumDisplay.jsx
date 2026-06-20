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

import { useEffect, useRef, useState } from 'react'
import { fetchAudioBands } from '../api.js'

const LABELS = ['Bass', 'Lo-Mid', 'Mid', 'Hi-Mid', 'Treble']

// Interpolate color from green (0) to red (1) via yellow
function bandColor(energy) {
  const e = Math.max(0, Math.min(1, energy))
  // green → yellow → red
  if (e < 0.5) {
    const t = e * 2
    const r = Math.round(t * 255)
    return `rgb(${r},200,0)`
  } else {
    const t = (e - 0.5) * 2
    const g = Math.round((1 - t) * 200)
    return `rgb(255,${g},0)`
  }
}

export default function SpectrumDisplay() {
  const canvasRef = useRef(null)
  const [unavailable, setUnavailable] = useState(false)
  const beatTimerRef = useRef(null)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      if (cancelled) return
      try {
        const data = await fetchAudioBands()
        setUnavailable(false)
        const canvas = canvasRef.current
        if (!canvas) return
        const ctx = canvas.getContext('2d')
        const w = canvas.width
        const h = canvas.height
        ctx.clearRect(0, 0, w, h)

        const bands = data.bands || []
        const count = bands.length
        const barW = Math.floor(w / count)
        const labelH = 14
        const barAreaH = h - labelH

        bands.forEach((energy, i) => {
          const barH = Math.round(energy * barAreaH)
          const x = i * barW
          const y = barAreaH - barH

          ctx.fillStyle = bandColor(energy)
          ctx.fillRect(x + 1, y, barW - 2, barH)

          // label
          ctx.fillStyle = '#9e9e9e'
          ctx.font = '9px Consolas, monospace'
          ctx.textAlign = 'center'
          ctx.fillText(LABELS[i] || '', x + barW / 2, h - 2)
        })

        if (data.beat) {
          canvas.style.borderColor = '#ffffff'
          if (beatTimerRef.current) clearTimeout(beatTimerRef.current)
          beatTimerRef.current = setTimeout(() => {
            if (canvasRef.current) canvasRef.current.style.borderColor = ''
          }, 80)
        }
      } catch {
        if (!cancelled) setUnavailable(true)
      }
    }

    const id = setInterval(poll, 100)
    poll()

    return () => {
      cancelled = true
      clearInterval(id)
      if (beatTimerRef.current) clearTimeout(beatTimerRef.current)
    }
  }, [])

  if (unavailable) {
    return (
      <div className="spectrum-wrap" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text2)', fontSize: 11 }}>
        audio unavailable
      </div>
    )
  }

  return (
    <div className="spectrum-wrap">
      <canvas
        ref={canvasRef}
        width={500}
        height={60}
        style={{ width: '100%', height: '60px', display: 'block', border: '1px solid transparent', transition: 'border-color 0.05s' }}
      />
    </div>
  )
}
