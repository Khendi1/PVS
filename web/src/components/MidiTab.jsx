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
