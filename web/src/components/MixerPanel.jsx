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
