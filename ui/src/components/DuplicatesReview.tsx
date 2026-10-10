
import { useState, useEffect } from 'react';

export interface DuplicatePair {
  match_type: 'phash_exact' | 'clip_near';
  similarity: number;
  photo_a: {
    id: number;
    filepath: string;
    date_taken: string | null;
  };
  photo_b: {
    id: number;
    filepath: string;
    date_taken: string | null;
  };
}

interface DuplicatesReviewProps {
  apiBase?: string;
  onUpdateStats?: () => void;
}

export default function DuplicatesReview({
  apiBase = 'http://127.0.0.1:8000',
  onUpdateStats,
}: DuplicatesReviewProps) {
  const [pairs, setPairs] = useState<DuplicatePair[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [resolvingIdx, setResolvingIdx] = useState<number | null>(null);

  const fetchDuplicates = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${apiBase}/api/duplicates`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: DuplicatePair[] = await res.json();
      setPairs(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load duplicates');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDuplicates();
  }, [apiBase]);

  const handleResolve = async (index: number, _keepPhotoId: number | null, duplicatePhotoId: number | null) => {
    try {
      setResolvingIdx(index);
      if (duplicatePhotoId) {
        const res = await fetch(`${apiBase}/api/duplicates/resolve`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            duplicate_photo_id: duplicatePhotoId,
            action: 'mark_duplicate',
          }),
        });
        if (!res.ok) throw new Error(`Failed to resolve duplicate (HTTP ${res.status})`);
      }

      setPairs((prev) => prev.filter((_, idx) => idx !== index));
      if (onUpdateStats) onUpdateStats();
    } catch (err: any) {
      alert(`Error: ${err.message}`);
    } finally {
      setResolvingIdx(null);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', color: 'var(--text-secondary)' }}>
        Loading duplicates...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', color: '#FF3B30', textAlign: 'center' }}>
        <p>Error: {error}</p>
        <button
          onClick={fetchDuplicates}
          style={{
            marginTop: '8px',
            padding: '4px 12px',
            backgroundColor: 'var(--accent)',
            color: 'var(--text-on-accent)',
            border: 'none',
            borderRadius: '4px',
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (pairs.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', color: 'var(--text-secondary)', gap: '6px' }}>
        <span style={{ fontSize: '13px' }}>No duplicates found</span>
        <span style={{ fontSize: '11px' }}>
          Your photo library has no pending exact or near duplicate candidates.
        </span>
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', overflow: 'hidden' }}>
      {/* Native Desktop Toolbar */}
      <div
        style={{
          height: '38px',
          minHeight: '38px',
          borderBottom: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-app)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 12px',
        }}
      >
        <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-primary)' }}>
          Review Duplicates
        </span>
        <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
          {pairs.length} {pairs.length === 1 ? 'pair' : 'pairs'}
        </span>
      </div>

      {/* Comparison List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {pairs.map((pair, idx) => {
          const isResolving = resolvingIdx === idx;
          const isExact = pair.match_type === 'phash_exact';
          const filenameA = pair.photo_a.filepath.split(/[/\\]/).pop() || pair.photo_a.filepath;
          const filenameB = pair.photo_b.filepath.split(/[/\\]/).pop() || pair.photo_b.filepath;

          return (
            <div
              key={`${pair.photo_a.id}-${pair.photo_b.id}`}
              style={{
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                backgroundColor: 'var(--bg-surface)',
                overflow: 'hidden',
                opacity: isResolving ? 0.5 : 1,
              }}
            >
              {/* Item Top Bar */}
              <div
                style={{
                  height: '32px',
                  padding: '0 10px',
                  backgroundColor: 'var(--bg-sidebar)',
                  borderBottom: '1px solid var(--border-color)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  fontSize: '11px',
                }}
              >
                <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                  {isExact ? 'Exact duplicate (pHash match)' : `Near duplicate (${Math.round(pair.similarity * 100)}% CLIP similarity)`}
                </span>

                <div style={{ display: 'flex', gap: '6px' }}>
                  <button
                    disabled={isResolving}
                    onClick={() => handleResolve(idx, pair.photo_a.id, pair.photo_b.id)}
                    style={{
                      height: '22px',
                      padding: '0 8px',
                      backgroundColor: 'var(--accent)',
                      color: 'var(--text-on-accent)',
                      border: 'none',
                      borderRadius: '3px',
                      fontSize: '11px',
                      fontWeight: 500,
                    }}
                  >
                    Keep Left
                  </button>
                  <button
                    disabled={isResolving}
                    onClick={() => handleResolve(idx, pair.photo_b.id, pair.photo_a.id)}
                    style={{
                      height: '22px',
                      padding: '0 8px',
                      backgroundColor: 'var(--accent)',
                      color: 'var(--text-on-accent)',
                      border: 'none',
                      borderRadius: '3px',
                      fontSize: '11px',
                      fontWeight: 500,
                    }}
                  >
                    Keep Right
                  </button>
                  <button
                    disabled={isResolving}
                    onClick={() => handleResolve(idx, null, null)}
                    style={{
                      height: '22px',
                      padding: '0 8px',
                      backgroundColor: 'transparent',
                      color: 'var(--text-secondary)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '3px',
                      fontSize: '11px',
                    }}
                  >
                    Keep Both
                  </button>
                </div>
              </div>

              {/* Side-by-side comparison */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1px', backgroundColor: 'var(--border-color)' }}>
                {/* Photo A */}
                <div style={{ backgroundColor: 'var(--bg-surface)', padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div style={{ width: '100%', height: '240px', backgroundColor: '#000000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img
                      src={`${apiBase}/api/photos/${pair.photo_a.id}/file`}
                      alt={filenameA}
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                  </div>
                  <div style={{ fontSize: '11px' }}>
                    <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{filenameA}</span>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '10px' }}>
                      {pair.photo_a.date_taken || 'No date'} • {pair.photo_a.filepath}
                    </p>
                  </div>
                </div>

                {/* Photo B */}
                <div style={{ backgroundColor: 'var(--bg-surface)', padding: '10px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div style={{ width: '100%', height: '240px', backgroundColor: '#000000', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <img
                      src={`${apiBase}/api/photos/${pair.photo_b.id}/file`}
                      alt={filenameB}
                      style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
                    />
                  </div>
                  <div style={{ fontSize: '11px' }}>
                    <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{filenameB}</span>
                    <p style={{ color: 'var(--text-secondary)', marginTop: '2px', fontSize: '10px' }}>
                      {pair.photo_b.date_taken || 'No date'} • {pair.photo_b.filepath}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
