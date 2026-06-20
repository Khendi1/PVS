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

import { useState, useRef, useEffect } from 'react'
import { midiLearnStart, midiLearnCancel, midiLearnStatus } from '../api.js'

export default function MidiTab({ params }) {
  const [selectedParam, setSelectedParam] = useState('')
  const [learning, setLearning] = useState(false)
  const [status, setStatus] = useState('')
  const pollRef = useRef(null)

  // Group params for the select optgroups
  const groups = {}
  for (const p of params) {
    const g = p.group || 'Other'
    if (!groups[g]) groups[g] = []
    groups[g].push(p.name)
  }

  async function handleLearn() {
    if (!selectedParam) { setStatus('Select a param first.'); return }
    try {
      await midiLearnStart(selectedParam)
      setLearning(true)
      setStatus(`Waiting for CC... move a knob mapped to: ${selectedParam}`)
      pollRef.current = setInterval(async () => {
        try {
          const s = await midiLearnStatus()
          if (!s.learning) {
            stopPoll('Mapped!')
          }
        } catch (e) { /* ignore */ }
      }, 500)
    } catch (e) {
      setStatus(`Error: ${e.message}`)
    }
  }

  async function handleCancel() {
    try { await midiLearnCancel() } catch (e) { /* ignore */ }
    stopPoll('Cancelled.')
  }

  function stopPoll(msg) {
    clearInterval(pollRef.current)
    pollRef.current = null
    setLearning(false)
    setStatus(msg)
    setTimeout(() => setStatus(''), 3000)
  }

  useEffect(() => () => clearInterval(pollRef.current), [])

  return (
    <div className="midi-tab">
      <div className="midi-group-box">
        <h3>MIDI Learn</h3>
        <div className="midi-row">
          <select
            className="midi-select"
            value={selectedParam}
            onChange={e => setSelectedParam(e.target.value)}
          >
            <option value="">— select param —</option>
            {Object.entries(groups).map(([gName, names]) => (
              <optgroup key={gName} label={gName}>
                {names.map(n => (
                  <option key={n} value={n}>{n.replace(/_/g, ' ')}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <div className="midi-row" style={{ marginTop: 6 }}>
          <button
            className={learning ? 'btn-active' : 'btn-idle'}
            onClick={handleLearn}
            disabled={learning}
          >
            Learn
          </button>
          <button onClick={handleCancel}>Cancel</button>
        </div>
        <div className="midi-status">{status}</div>
      </div>
    </div>
  )
}
