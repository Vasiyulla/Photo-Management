import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';

export type BackendStatus = 'INITIALIZING' | 'CONNECTING' | 'READY' | 'ERROR';

interface BackendContextValue {
  status: BackendStatus;
  apiBase: string;
  error: string | null;
  retry: () => void;
}

const BackendContext = createContext<BackendContextValue>({
  status: 'INITIALIZING',
  apiBase: 'http://127.0.0.1:8000',
  error: null,
  retry: () => {},
});

export const useBackend = () => useContext(BackendContext);

interface BackendProviderProps {
  children: React.ReactNode;
}

export const BackendProvider: React.FC<BackendProviderProps> = ({ children }) => {
  const [status, setStatus] = useState<BackendStatus>('INITIALIZING');
  const [apiBase, setApiBase] = useState<string>('http://127.0.0.1:8000');
  const [error, setError] = useState<string | null>(null);
  const [progressMsg, setProgressMsg] = useState<string>('Starting local AI engine...');

  const checkHealth = async (baseUrl: string): Promise<boolean> => {
    try {
      const res = await fetch(`${baseUrl}/api/health`, { method: 'GET' });
      if (res.ok) return true;
    } catch {
      // Endpoint not up yet
    }
    try {
      const res = await fetch(`${baseUrl}/api/persons`, { method: 'GET' });
      if (res.ok) return true;
    } catch {
      // Endpoint not up yet
    }
    return false;
  };

  const initBackend = useCallback(async () => {
    setStatus('CONNECTING');
    setError(null);
    setProgressMsg('Connecting to local AI engine...');

    const isTauri = typeof window !== 'undefined' && Boolean((window as any).__TAURI__);
    let targetUrl = 'http://127.0.0.1:8000';

    const maxAttempts = 50; // 50 * 300ms = 15 seconds
    let attempts = 0;

    const poll = async () => {
      attempts++;

      if (isTauri) {
        try {
          const tauriInvoke = (window as any).__TAURI__.invoke;
          const statusObj = await tauriInvoke('get_backend_status');
          if (statusObj && statusObj.ready && statusObj.url) {
            targetUrl = statusObj.url;
            setApiBase(targetUrl);
            const healthy = await checkHealth(targetUrl);
            if (healthy) {
              setStatus('READY');
              return;
            }
          } else if (statusObj && statusObj.error) {
            setError(`Engine error: ${statusObj.error}`);
            setStatus('ERROR');
            return;
          }
        } catch {
          // Tauri IPC not ready yet
        }
      } else {
        // Browser dev fallback
        const healthy = await checkHealth(targetUrl);
        if (healthy) {
          setApiBase(targetUrl);
          setStatus('READY');
          return;
        }
      }

      if (attempts >= maxAttempts) {
        setError('PhotoVault AI Engine took too long to respond. Click Retry to reconnect.');
        setStatus('ERROR');
        return;
      }

      if (attempts > 5) {
        setProgressMsg('Loading AI models (OpenCLIP, ArcFace, SCRFD)...');
      }

      setTimeout(poll, 300);
    };

    poll();
  }, []);

  useEffect(() => {
    initBackend();
  }, [initBackend]);

  if (status !== 'READY') {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: '#0F172A',
        color: '#F8FAFC',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        zIndex: 9999,
        padding: '24px',
        userSelect: 'none',
      }}>
        {/* Logo / Shield Emblem */}
        <div style={{
          width: '72px',
          height: '72px',
          borderRadius: '20px',
          background: 'linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          boxShadow: '0 0 40px rgba(79, 70, 229, 0.4)',
          marginBottom: '28px',
          animation: status === 'ERROR' ? 'none' : 'pulse 2s infinite ease-in-out',
        }}>
          <svg width="38" height="38" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            <circle cx="12" cy="11" r="3"/>
          </svg>
        </div>

        <h1 style={{ fontSize: '24px', fontWeight: 700, margin: '0 0 8px 0', letterSpacing: '-0.5px' }}>
          PhotoVault
        </h1>

        <p style={{ fontSize: '14px', color: '#94A3B8', margin: '0 0 24px 0' }}>
          Local Offline Face-Aware Photo Manager
        </p>

        {status === 'ERROR' ? (
          <div style={{
            background: 'rgba(239, 68, 68, 0.1)',
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '8px',
            padding: '16px 20px',
            maxWidth: '420px',
            textAlign: 'center',
          }}>
            <p style={{ color: '#F87171', fontSize: '13px', margin: '0 0 14px 0', lineHeight: '1.5' }}>
              {error}
            </p>
            <button
              onClick={initBackend}
              style={{
                backgroundColor: '#4F46E5',
                color: '#FFFFFF',
                border: 'none',
                borderRadius: '6px',
                padding: '8px 18px',
                fontSize: '13px',
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Retry Connection
            </button>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div style={{
              width: '200px',
              height: '4px',
              backgroundColor: 'rgba(255,255,255,0.1)',
              borderRadius: '2px',
              overflow: 'hidden',
              marginBottom: '14px',
            }}>
              <div style={{
                width: '60%',
                height: '100%',
                background: 'linear-gradient(90deg, #4F46E5, #06B6D4)',
                borderRadius: '2px',
                animation: 'progress 1.5s infinite ease-in-out',
              }} />
            </div>
            <p style={{ fontSize: '13px', color: '#64748B', margin: 0 }}>
              {progressMsg}
            </p>
          </div>
        )}

        <style>{`
          @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.85; }
          }
          @keyframes progress {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(200%); }
          }
        `}</style>
      </div>
    );
  }

  return (
    <BackendContext.Provider value={{ status, apiBase, error, retry: initBackend }}>
      {children}
    </BackendContext.Provider>
  );
};
