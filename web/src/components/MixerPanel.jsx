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

import { patchAction } from '../api.js'
import SubgroupPanel from './SubgroupPanel.jsx'

export default function MixerPanel({ params, onPatchLoad }) {
  async function handlePatch(action) {
    try {
      await patchAction(action)
      if (action !== 'save' && onPatchLoad) onPatchLoad()
    } catch (e) { /* ignore */ }
  }

  return (
    <div className="mixer-panel">
      <div className="patch-row">
        <button onClick={() => handlePatch('save')}>Save</button>
        <button onClick={() => handlePatch('prev')}>← Prev</button>
        <button onClick={() => handlePatch('random')}>Random</button>
        <button onClick={() => handlePatch('next')}>Next →</button>
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <SubgroupPanel params={params} />
      </div>
    </div>
  )
}
