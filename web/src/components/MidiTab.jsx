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
import {
  midiLearnStart,
  midiLearnCancel,
  midiLearnStatus,
  exportMidiMappings,
  importMidiMappings,
} from '../api.js'

export default function MidiTab({ params }) {
  const [selectedParam, setSelectedParam] = useState('')
  const [learning, setLearning] = useState(false)
  const [status, setStatus] = useState('')
  const [shareStatus, setShareStatus] = useState('')
  const pollRef = useRef(null)
  const fileRef = useRef(null)

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

  async function handleExport() {
    try {
      const text = await exportMidiMappings()
      const blob = new Blob([text], { type: 'application/x-yaml' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'midi_mappings.yaml'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      setShareStatus('Exported midi_mappings.yaml')
      setTimeout(() => setShareStatus(''), 3000)
    } catch (e) {
      setShareStatus(`Export error: ${e.message}`)
    }
  }

  async function handleImport(e) {
    const file = e.target.files && e.target.files[0]
    if (!file) return
    try {
      const text = await file.text()
      const res = await importMidiMappings(text)
      setShareStatus(`Imported ${res.mappings} mapping(s) across ${res.ports} port(s).`)
      setTimeout(() => setShareStatus(''), 3000)
    } catch (e) {
      setShareStatus(`Import error: ${e.message}`)
    } finally {
      // Reset so the same file can be re-selected to re-import.
      if (fileRef.current) fileRef.current.value = ''
    }
  }

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

      <div className="midi-group-box">
        <h3>Share Mappings</h3>
        <div className="midi-row">
          <button onClick={handleExport}>Export</button>
          <button onClick={() => fileRef.current && fileRef.current.click()}>
            Import
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".yaml,.yml,application/x-yaml,text/yaml"
            style={{ display: 'none' }}
            onChange={handleImport}
          />
        </div>
        <div className="midi-status">{shareStatus}</div>
      </div>
    </div>
  )
}
