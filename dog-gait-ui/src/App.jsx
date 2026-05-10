import { useState, useRef, useCallback } from 'react'
import { Download, RotateCcw } from 'lucide-react'
import UploadZone from './components/UploadZone'
import ProgressBar from './components/ProgressBar'
import GaitReport from './components/GaitReport'

const POLL_MS = 1200

export default function App() {
  const [file,     setFile]     = useState(null)
  const [jobId,    setJobId]    = useState(null)
  const [status,   setStatus]   = useState(null)   // queued|processing|done|error
  const [progress, setProgress] = useState(0)
  const [summary,  setSummary]  = useState(null)
  const [error,    setError]    = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)

  const pollRef = useRef(null)

  // ── helpers ──────────────────────────────────────────────────────────────────
  function stopPoll() {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  function reset() {
    stopPoll()
    setFile(null); setJobId(null); setStatus(null)
    setProgress(0); setSummary(null); setError(null)
    if (videoUrl) { URL.revokeObjectURL(videoUrl); setVideoUrl(null) }
  }

  // ── upload & start ────────────────────────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    if (!file) return
    setError(null); setStatus('queued'); setProgress(0); setSummary(null); setVideoUrl(null)

    const fd = new FormData()
    fd.append('file', file)

    let id
    try {
      const res = await fetch('/upload', { method: 'POST', body: fd })
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Upload failed') }
      const data = await res.json()
      id = data.job_id
      setJobId(id)
    } catch (e) {
      setError(e.message); setStatus('error'); return
    }

    // poll status
    pollRef.current = setInterval(async () => {
      try {
        const res  = await fetch(`/status/${id}`)
        const data = await res.json()
        setStatus(data.status)
        setProgress(data.progress ?? 0)

        if (data.status === 'error') {
          stopPoll(); setError(data.error || 'Processing failed')
        }

        if (data.status === 'done') {
          stopPoll()
          // fetch summary
          const sr = await fetch(`/summary/${id}`)
          const sm = await sr.json()
          setSummary(sm)
          // build video URL
          setVideoUrl(`/result/${id}`)
        }
      } catch (_) { /* network hiccup — keep polling */ }
    }, POLL_MS)
  }, [file])

  const busy = status === 'queued' || status === 'processing'
  const done = status === 'done'

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>

      {/* ── Header ── */}
      <header style={{
        background: 'var(--surface)',
        borderBottom: '1px solid var(--border)',
        padding: '16px 32px',
        display: 'flex', alignItems: 'center', gap: 14,
        position: 'sticky', top: 0, zIndex: 100,
      }}>
        <span style={{ fontSize: 28 }}>🐾</span>
        <div>
          <h1 style={{
            fontSize: 18, fontWeight: 800,
            background: 'linear-gradient(90deg, #58a6ff, #bc8cff)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>
            Dog Gait Analysis
          </h1>
          <p style={{ fontSize: 12, color: 'var(--muted)', marginTop: 1 }}>
            YOLOv8 Pose · 24 Keypoints · Limping Detection
          </p>
        </div>
        {(done || error) && (
          <button
            onClick={reset}
            style={{
              marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6,
              background: 'var(--surface2)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius-sm)', padding: '7px 14px',
              color: 'var(--text)', cursor: 'pointer', fontSize: 13,
            }}
          >
            <RotateCcw size={14} /> New Analysis
          </button>
        )}
      </header>

      {/* ── Body ── */}
      <main style={{
        flex: 1, maxWidth: 1200, width: '100%',
        margin: '0 auto', padding: '32px 24px',
        display: 'flex', flexDirection: 'column', gap: 28,
      }}>

        {/* ── Upload card (hide when done) ── */}
        {!done && (
          <section style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 'var(--radius)', padding: '28px 32px',
          }}>
            <SectionTitle>Upload Video</SectionTitle>
            <UploadZone
              file={file}
              onFile={setFile}
              onClear={() => setFile(null)}
              disabled={busy}
            />

            {error && (
              <div style={{
                marginTop: 14, padding: '12px 16px',
                background: '#f8514920', border: '1px solid var(--red)',
                borderRadius: 'var(--radius-sm)', color: 'var(--red)', fontSize: 13,
              }}>
                ⚠ {error}
              </div>
            )}

            {status && status !== 'error' && (
              <div style={{ marginTop: 20 }}>
                <ProgressBar pct={progress} status={status} />
              </div>
            )}

            {!busy && (
              <button
                onClick={handleSubmit}
                disabled={!file}
                style={{
                  marginTop: 20, width: '100%', padding: '13px',
                  background: file
                    ? 'linear-gradient(135deg, #1f6feb, #8957e5)'
                    : 'var(--surface2)',
                  border: 'none', borderRadius: 'var(--radius-sm)',
                  color: file ? '#fff' : 'var(--muted)',
                  fontSize: 15, fontWeight: 700, cursor: file ? 'pointer' : 'not-allowed',
                  transition: 'opacity 0.2s',
                }}
              >
                Run Gait Analysis
              </button>
            )}
          </section>
        )}

        {/* ── Results ── */}
        {done && summary && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>

            {/* Video + download */}
            <section style={{
              background: 'var(--surface)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius)', padding: '28px 32px',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <SectionTitle>Processed Video</SectionTitle>
                <a
                  href={videoUrl}
                  download={`dog_gait_${jobId?.slice(0, 8)}.mp4`}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 6,
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)', padding: '7px 14px',
                    color: 'var(--accent)', textDecoration: 'none', fontSize: 13, fontWeight: 600,
                  }}
                >
                  <Download size={14} /> Download
                </a>
              </div>
              <div style={{
                background: '#000', borderRadius: 'var(--radius-sm)',
                overflow: 'hidden', aspectRatio: '16/9',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <video
                  src={videoUrl}
                  controls
                  autoPlay
                  loop
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                />
              </div>
            </section>

            {/* Gait report */}
            <section style={{
              background: 'var(--surface)', border: '1px solid var(--border)',
              borderRadius: 'var(--radius)', padding: '28px 32px',
            }}>
              <SectionTitle style={{ marginBottom: 20 }}>Gait Analysis Report</SectionTitle>
              <GaitReport summary={summary} />
            </section>

          </div>
        )}

      </main>
    </div>
  )
}

function SectionTitle({ children, style }) {
  return (
    <h2 style={{
      fontSize: 12, fontWeight: 700, textTransform: 'uppercase',
      letterSpacing: '0.08em', color: 'var(--muted)', marginBottom: 18,
      ...style,
    }}>
      {children}
    </h2>
  )
}
