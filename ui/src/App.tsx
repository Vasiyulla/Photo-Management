import { useState, useEffect, useRef } from 'react';
import PeopleGrid from './components/PeopleGrid';
import PhotoGrid from './components/PhotoGrid';
import DuplicatesReview from './components/DuplicatesReview';
import { useBackend } from './context/BackendContext';

export type TabType = 'photos' | 'people' | 'duplicates';

export default function App() {
  const { apiBase } = useBackend();
  const [activeTab, setActiveTab] = useState<TabType>(() => {
    const tabParam = new URLSearchParams(window.location.search).get('tab');
    if (tabParam === 'people' || tabParam === 'duplicates' || tabParam === 'photos') return tabParam;
    return 'photos';
  });
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [engineOnline, setEngineOnline] = useState<boolean>(false);
  const [filterPersonId, setFilterPersonId] = useState<number | null>(null);

  // Counts for sidebar
  const [photoCount, setPhotoCount] = useState<number>(0);
  const [peopleCount, setPeopleCount] = useState<number>(0);
  const [duplicatesCount, setDuplicatesCount] = useState<number>(0);

  // Scan modal state
  const [showScanModal, setShowScanModal] = useState<boolean>(false);
  const [scanPath, setScanPath] = useState<string>('');
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [scanStatus, setScanStatus] = useState<string | null>(null);

  // Import state
  const [isImporting, setIsImporting] = useState<boolean>(false);
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const [showImportModal, setShowImportModal] = useState<boolean>(false);
  const [importManualPaths, setImportManualPaths] = useState<string>('');
  const [importReport, setImportReport] = useState<{
    imported: number;
    skipped_duplicates: number;
    failed: number;
    skipped_details?: string[];
    failed_details?: { path: string; error: string }[];
  } | null>(null);

  const [refreshKey, setRefreshKey] = useState<number>(0);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  // Fetch summary counts for native sidebar
  const fetchStats = async () => {
    try {
      const [pRes, peoRes, dupRes] = await Promise.all([
        fetch(`${apiBase}/api/photos?limit=10000`),
        fetch(`${apiBase}/api/persons`),
        fetch(`${apiBase}/api/duplicates`),
      ]);

      if (pRes.ok) {
        setEngineOnline(true);
        // We can get total photos count from a query or photo headers
        const pData = await pRes.json();
        // If query returns list, we know engine is up
        if (Array.isArray(pData)) setPhotoCount(pData.length);
      }
      if (peoRes.ok) {
        const peoData = await peoRes.json();
        if (Array.isArray(peoData)) setPeopleCount(peoData.length);
      }
      if (dupRes.ok) {
        const dupData = await dupRes.json();
        if (Array.isArray(dupData)) setDuplicatesCount(dupData.length);
      }
    } catch {
      setEngineOnline(false);
    }
  };

  useEffect(() => {
    fetchStats();
  }, [refreshKey]);

  const handleSelectPerson = (personId: number) => {
    setFilterPersonId(personId);
    setActiveTab('photos');
  };

  const handleTabChange = (tab: TabType) => {
    if (tab !== 'photos') {
      setFilterPersonId(null);
    }
    setActiveTab(tab);
  };

  // Polling for background scan/import tasks
  const pollTimerRef = useRef<any>(null);

  const startStatusPolling = () => {
    if (pollTimerRef.current) return;
    setIsScanning(true);
    pollTimerRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${apiBase}/api/scan/status`);
        if (!res.ok) return;
        const task = await res.json();
        if (task.status === 'running') {
          setIsScanning(true);
          setScanStatus(task.message || 'Indexing photos in background...');
          fetchStats();
        } else if (task.status === 'completed') {
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          setIsScanning(false);
          setScanStatus(task.message || 'Indexing completed successfully.');
          setRefreshKey((k) => k + 1);
          fetchStats();
          setTimeout(() => {
            setScanStatus(null);
          }, 3500);
        } else if (task.status === 'error') {
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          setIsScanning(false);
          setScanStatus(`Indexing error: ${task.error || 'Unknown error'}`);
          setTimeout(() => {
            setScanStatus(null);
          }, 6000);
        } else if (task.status === 'idle') {
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
          setIsScanning(false);
        }
      } catch (err) {
        console.warn('Scan status poll error:', err);
      }
    }, 800);
  };

  // Check if an indexing task was already running on boot
  useEffect(() => {
    fetch(`${apiBase}/api/scan/status`)
      .then((r) => (r.ok ? r.json() : null))
      .then((task) => {
        if (task && task.status === 'running') {
          startStatusPolling();
        }
      })
      .catch(() => {});

    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, [apiBase]);

  const handleStartScan = async () => {
    const cleaned = scanPath.trim().replace(/^["']|["']$/g, '');
    if (!cleaned) return;

    try {
      setIsScanning(true);
      setScanStatus('Scanning folder...');
      const res = await fetch(`${apiBase}/api/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder_path: cleaned }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to scan folder');

      if (data.status === 'started') {
        setScanStatus(`Found ${data.new_photos} new photos. Indexing in background...`);
        setTimeout(() => {
          setShowScanModal(false);
          setScanPath('');
          setRefreshKey((k) => k + 1);
          fetchStats();
        }, 800);
        startStatusPolling();
      } else {
        setScanStatus(data.message || 'All photos are up to date.');
        setTimeout(() => {
          setShowScanModal(false);
          setScanStatus(null);
          setScanPath('');
          setRefreshKey((k) => k + 1);
          fetchStats();
        }, 1200);
      }
    } catch (err: any) {
      setScanStatus(`Error: ${err.message}`);
      setIsScanning(false);
    }
  };

  const executeImport = async (filePaths: string[]) => {
    if (!filePaths || filePaths.length === 0) return;
    try {
      setIsImporting(true);
      setImportStatus(`Registering ${filePaths.length} image${filePaths.length === 1 ? '' : 's'}...`);
      const res = await fetch(`${apiBase}/api/import`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_paths: filePaths }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to import images');

      setImportReport({
        imported: data.imported,
        skipped_duplicates: data.skipped_duplicates,
        failed: data.failed,
        skipped_details: data.skipped_details,
        failed_details: data.failed_details,
      });
      setRefreshKey((k) => k + 1);
      fetchStats();

      if (data.status === 'started') {
        startStatusPolling();
      }
    } catch (err: any) {
      alert(`Import error: ${err.message}`);
    } finally {
      setIsImporting(false);
      setImportStatus(null);
    }
  };

  const handleTriggerNativeDialog = async () => {
    const tauriDialog = (window as any).__TAURI__?.dialog;
    if (tauriDialog?.open) {
      try {
        const selected = await tauriDialog.open({
          multiple: true,
          filters: [
            {
              name: 'Images',
              extensions: ['jpg', 'jpeg', 'png', 'webp', 'jfif', 'bmp', 'tiff', 'tif', 'avif'],
            },
          ],
        });
        if (!selected) return;
        const paths = Array.isArray(selected) ? selected : [selected];
        if (paths.length > 0) {
          setShowImportModal(false);
          await executeImport(paths);
        }
      } catch (err: any) {
        console.warn('Tauri dialog error:', err);
      }
    } else {
      try {
        const res = await fetch(`${apiBase}/api/dialog/pick-images`);
        if (res.ok) {
          const data = await res.json();
          if (data.file_paths && data.file_paths.length > 0) {
            setShowImportModal(false);
            await executeImport(data.file_paths);
          }
        }
      } catch (err: any) {
        console.warn('Native dialog endpoint error:', err);
      }
    }
  };

  const handleImportClick = async () => {
    const tauriDialog = (window as any).__TAURI__?.dialog;
    if (tauriDialog?.open) {
      try {
        const selected = await tauriDialog.open({
          multiple: true,
          filters: [
            {
              name: 'Images',
              extensions: ['jpg', 'jpeg', 'png', 'webp', 'jfif', 'bmp', 'tiff', 'tif', 'avif'],
            },
          ],
        });
        if (!selected) return;
        const paths = Array.isArray(selected) ? selected : [selected];
        if (paths.length > 0) {
          await executeImport(paths);
        }
      } catch (err: any) {
        console.warn('Tauri dialog failed, opening manual modal:', err);
        setShowImportModal(true);
      }
    } else {
      // In web dev browser mode: try native file dialog first, or open modal
      try {
        const res = await fetch(`${apiBase}/api/dialog/pick-images`);
        if (res.ok) {
          const data = await res.json();
          if (data.file_paths && data.file_paths.length > 0) {
            await executeImport(data.file_paths);
            return;
          } else if (data.file_paths && data.file_paths.length === 0) {
            // User cancelled file picker
            return;
          }
        }
      } catch {
        // Fall back to modal
      }
      setShowImportModal(true);
    }
  };

  return (
    <div style={{ display: 'flex', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      {/* Persistent Left Sidebar */}
      <aside
        style={{
          width: '210px',
          minWidth: '210px',
          maxWidth: '210px',
          backgroundColor: 'var(--bg-sidebar)',
          borderRight: '1px solid var(--border-color)',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          userSelect: 'none',
        }}
      >
        {/* Top: App Header & Navigation */}
        <div>
          {/* Native Window Title / App Name */}
          <div
            style={{
              height: '44px',
              minHeight: '44px',
              maxHeight: '44px',
              padding: '0 14px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              borderBottom: '1px solid var(--border-color)',
              boxSizing: 'border-box',
            }}
          >
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              PhotoVault
            </span>
            <span
              title={engineOnline ? 'Core Engine Online' : 'Core Engine Offline'}
              style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: engineOnline ? '#34C759' : '#FF9500',
              }}
            />
          </div>

          {/* Navigation Section: Library */}
          <div style={{ padding: '10px 8px 4px 8px' }}>
            <div
              style={{
                padding: '4px 8px',
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                letterSpacing: '0.3px',
              }}
            >
              LIBRARY
            </div>

            <button
              onClick={() => handleTabChange('photos')}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                borderRadius: '5px',
                border: 'none',
                backgroundColor: activeTab === 'photos' ? 'var(--bg-selected)' : 'transparent',
                color: activeTab === 'photos' ? 'var(--text-on-accent)' : 'var(--text-primary)',
                fontWeight: activeTab === 'photos' ? 600 : 400,
                fontSize: '13px',
                textAlign: 'left',
                marginTop: '2px',
              }}
            >
              <span>Photos</span>
              {photoCount > 0 && (
                <span style={{ fontSize: '11px', opacity: activeTab === 'photos' ? 0.9 : 0.6, whiteSpace: 'nowrap', flexShrink: 0 }}>
                  {photoCount}
                </span>
              )}
            </button>

            <button
              onClick={() => handleTabChange('people')}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                borderRadius: '5px',
                border: 'none',
                backgroundColor: activeTab === 'people' ? 'var(--bg-selected)' : 'transparent',
                color: activeTab === 'people' ? 'var(--text-on-accent)' : 'var(--text-primary)',
                fontWeight: activeTab === 'people' ? 600 : 400,
                fontSize: '13px',
                textAlign: 'left',
                marginTop: '2px',
              }}
            >
              <span>People</span>
              {peopleCount > 0 && (
                <span style={{ fontSize: '11px', opacity: activeTab === 'people' ? 0.9 : 0.6 }}>
                  {peopleCount}
                </span>
              )}
            </button>

            <button
              onClick={() => handleTabChange('duplicates')}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '6px 10px',
                borderRadius: '5px',
                border: 'none',
                backgroundColor: activeTab === 'duplicates' ? 'var(--bg-selected)' : 'transparent',
                color: activeTab === 'duplicates' ? 'var(--text-on-accent)' : 'var(--text-primary)',
                fontWeight: activeTab === 'duplicates' ? 600 : 400,
                fontSize: '13px',
                textAlign: 'left',
                marginTop: '2px',
              }}
            >
              <span>Duplicates</span>
              {duplicatesCount > 0 && (
                <span style={{ fontSize: '11px', opacity: activeTab === 'duplicates' ? 0.9 : 0.6 }}>
                  {duplicatesCount}
                </span>
              )}
            </button>
          </div>

          {/* Navigation Section: Actions */}
          <div style={{ padding: '12px 8px 4px 8px' }}>
            <div
              style={{
                padding: '4px 8px',
                fontSize: '11px',
                fontWeight: 600,
                color: 'var(--text-secondary)',
                letterSpacing: '0.3px',
              }}
            >
              ACTIONS
            </div>

            <button
              disabled={isScanning || isImporting}
              onClick={() => setShowScanModal(true)}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 10px',
                borderRadius: '5px',
                border: 'none',
                backgroundColor: 'transparent',
                color: 'var(--text-primary)',
                fontSize: '12px',
                textAlign: 'left',
                marginTop: '2px',
                cursor: isScanning || isImporting ? 'not-allowed' : 'pointer',
                opacity: isScanning || isImporting ? 0.6 : 1,
              }}
            >
              <span>📁</span> Scan Folder...
            </button>

            <button
              disabled={isScanning || isImporting}
              onClick={handleImportClick}
              style={{
                width: '100%',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                padding: '6px 10px',
                borderRadius: '5px',
                border: 'none',
                backgroundColor: 'transparent',
                color: 'var(--text-primary)',
                fontSize: '12px',
                textAlign: 'left',
                marginTop: '3px',
                cursor: isScanning || isImporting ? 'not-allowed' : 'pointer',
                opacity: isScanning || isImporting ? 0.6 : 1,
              }}
            >
              <span>🖼️</span> Import Images...
            </button>

            {(isScanning || isImporting || scanStatus) && (
              <div
                style={{
                  margin: '8px 2px 0 2px',
                  padding: '8px 10px',
                  backgroundColor: 'var(--bg-input)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '4px',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', fontWeight: 600, color: 'var(--accent)' }}>
                  <span style={{ fontSize: '12px' }}>{isScanning || isImporting ? '⚡' : '✓'}</span>
                  <span>{isScanning ? 'Indexing Library' : isImporting ? 'Importing Images' : 'Index Status'}</span>
                </div>
                <div
                  title={scanStatus || importStatus || 'Processing in place...'}
                  style={{ fontSize: '10px', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
                >
                  {scanStatus || importStatus || 'Processing in background...'}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Bottom: Settings & Theme */}
        <div
          style={{
            padding: '10px 12px',
            borderTop: '1px solid var(--border-color)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Appearance</span>
          <button
            onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
            style={{
              padding: '3px 8px',
              borderRadius: '4px',
              border: '1px solid var(--border-color)',
              backgroundColor: 'var(--bg-app)',
              color: 'var(--text-primary)',
              fontSize: '11px',
            }}
          >
            {theme === 'light' ? 'Dark' : 'Light'}
          </button>
        </div>
      </aside>

      {/* Main Content Pane */}
      <main
        style={{
          flex: 1,
          height: '100vh',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          backgroundColor: 'var(--bg-app)',
        }}
      >
        {activeTab === 'photos' && (
          <PhotoGrid
            key={`${filterPersonId ?? 'all'}-${refreshKey}`}
            apiBase={apiBase}
            initialPersonId={filterPersonId}
            onUpdateStats={fetchStats}
          />
        )}
        {activeTab === 'people' && (
          <PeopleGrid
            key={refreshKey}
            apiBase={apiBase}
            onSelectPerson={handleSelectPerson}
            onUpdateStats={fetchStats}
          />
        )}
        {activeTab === 'duplicates' && (
          <DuplicatesReview key={refreshKey} apiBase={apiBase} onUpdateStats={fetchStats} />
        )}
      </main>

      {/* Native Scan Folder Modal */}
      {showScanModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.45)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={() => !isScanning && setShowScanModal(false)}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '20px',
              width: '420px',
              maxWidth: '90%',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Scan Folder
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
              Photos will be indexed in place on disk. No files are copied or moved.
            </p>

            <input
              type="text"
              autoFocus
              disabled={isScanning}
              placeholder="e.g. C:\Users\vasiy\Pictures"
              value={scanPath}
              onChange={(e) => setScanPath(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleStartScan()}
              style={{
                width: '100%',
                padding: '6px 10px',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                outline: 'none',
              }}
            />

            {scanStatus && (
              <div
                style={{
                  fontSize: '11px',
                  color: scanStatus.includes('Error') ? '#FF3B30' : 'var(--accent)',
                }}
              >
                {scanStatus}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '4px' }}>
              <button
                disabled={isScanning}
                onClick={() => setShowScanModal(false)}
                style={{
                  padding: '5px 12px',
                  backgroundColor: 'transparent',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                }}
              >
                Cancel
              </button>
              <button
                disabled={isScanning || !scanPath.trim()}
                onClick={handleStartScan}
                style={{
                  padding: '5px 14px',
                  backgroundColor: 'var(--accent)',
                  color: 'var(--text-on-accent)',
                  border: 'none',
                  borderRadius: '4px',
                  fontWeight: 500,
                  opacity: isScanning || !scanPath.trim() ? 0.6 : 1,
                }}
              >
                {isScanning ? 'Scanning...' : 'Scan'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import Images Modal */}
      {showImportModal && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.45)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={() => !isImporting && setShowImportModal(false)}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-color)',
              borderRadius: '6px',
              padding: '20px',
              width: '460px',
              maxWidth: '90%',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-primary)' }}>
              Import Images
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
              Select or specify individual image files to index in place. All files remain in their original directories without being moved or copied.
            </p>

            <button
              disabled={isImporting}
              onClick={handleTriggerNativeDialog}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '8px 12px',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px dashed var(--border-color)',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '12px',
                fontWeight: 500,
              }}
            >
              <span>📂</span> Choose with Native File Dialog...
            </button>

            <div style={{ fontSize: '11px', color: 'var(--text-secondary)', marginTop: '2px' }}>
              Or paste absolute file paths (one per line):
            </div>

            <textarea
              rows={4}
              disabled={isImporting}
              placeholder="C:\Users\vasiy\Pictures\sample.jpg&#10;C:\Users\vasiy\Pictures\family.png"
              value={importManualPaths}
              onChange={(e) => setImportManualPaths(e.target.value)}
              style={{
                width: '100%',
                padding: '8px',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '4px',
                fontFamily: 'monospace',
                fontSize: '11px',
                outline: 'none',
                resize: 'vertical',
              }}
            />

            {importStatus && (
              <div
                style={{
                  fontSize: '11px',
                  color: 'var(--accent)',
                }}
              >
                {importStatus}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '4px' }}>
              <button
                disabled={isImporting}
                onClick={() => setShowImportModal(false)}
                style={{
                  padding: '5px 12px',
                  backgroundColor: 'transparent',
                  color: 'var(--text-secondary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '4px',
                }}
              >
                Cancel
              </button>
              <button
                disabled={isImporting || !importManualPaths.trim()}
                onClick={() => {
                  const paths = importManualPaths
                    .split('\n')
                    .map((p) => p.trim().replace(/^["']|["']$/g, ''))
                    .filter(Boolean);
                  if (paths.length > 0) {
                    setShowImportModal(false);
                    setImportManualPaths('');
                    executeImport(paths);
                  }
                }}
                style={{
                  padding: '5px 14px',
                  backgroundColor: 'var(--accent)',
                  color: 'var(--text-on-accent)',
                  border: 'none',
                  borderRadius: '4px',
                  fontWeight: 500,
                  opacity: isImporting || !importManualPaths.trim() ? 0.6 : 1,
                }}
              >
                {isImporting ? 'Importing...' : 'Import Files'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Import Completion Summary Modal */}
      {importReport && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.45)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={() => setImportReport(null)}
        >
          <div
            style={{
              backgroundColor: 'var(--bg-surface)',
              border: '1px solid var(--border-color)',
              borderRadius: '8px',
              padding: '20px',
              width: '460px',
              maxWidth: '90%',
              display: 'flex',
              flexDirection: 'column',
              gap: '14px',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '18px' }}>🎉</span>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                Import Summary
              </div>
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '8px',
              }}
            >
              <div style={{ backgroundColor: 'var(--bg-card)', padding: '10px', borderRadius: '6px', textAlign: 'center', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '18px', fontWeight: 700, color: '#34C759' }}>{importReport.imported}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Photos Indexed</div>
              </div>
              <div style={{ backgroundColor: 'var(--bg-card)', padding: '10px', borderRadius: '6px', textAlign: 'center', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '18px', fontWeight: 700, color: 'var(--accent)' }}>{importReport.skipped_duplicates}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Duplicates Skipped</div>
              </div>
              <div style={{ backgroundColor: 'var(--bg-card)', padding: '10px', borderRadius: '6px', textAlign: 'center', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '18px', fontWeight: 700, color: importReport.failed > 0 ? '#FF3B30' : 'var(--text-secondary)' }}>{importReport.failed}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>Failed Files</div>
              </div>
            </div>

            {importReport.failed_details && importReport.failed_details.length > 0 && (
              <div style={{ marginTop: '2px' }}>
                <div style={{ fontSize: '11px', fontWeight: 600, color: '#FF3B30', marginBottom: '4px' }}>
                  Failed Files ({importReport.failed_details.length}):
                </div>
                <div
                  style={{
                    maxHeight: '110px',
                    overflowY: 'auto',
                    backgroundColor: 'var(--bg-input)',
                    borderRadius: '4px',
                    padding: '6px 8px',
                    fontSize: '10px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '4px',
                  }}
                >
                  {importReport.failed_details.map((f, i) => (
                    <div key={i} style={{ color: 'var(--text-primary)' }}>
                      <span style={{ fontWeight: 600 }}>{f.path.split(/[/\\]/).pop()}</span>: <span style={{ color: '#FF3B30' }}>{f.error}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {importReport.skipped_details && importReport.skipped_details.length > 0 && (
              <div>
                <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '4px' }}>
                  Skipped Duplicates ({importReport.skipped_details.length}):
                </div>
                <div
                  style={{
                    maxHeight: '70px',
                    overflowY: 'auto',
                    backgroundColor: 'var(--bg-input)',
                    borderRadius: '4px',
                    padding: '6px 8px',
                    fontSize: '10px',
                    color: 'var(--text-secondary)',
                  }}
                >
                  {importReport.skipped_details.map((p, i) => (
                    <div key={i} style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.split(/[/\\]/).pop()}
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '4px' }}>
              <button
                onClick={() => setImportReport(null)}
                style={{
                  padding: '5px 16px',
                  backgroundColor: 'var(--accent)',
                  color: 'var(--text-on-accent)',
                  border: 'none',
                  borderRadius: '4px',
                  fontWeight: 500,
                  fontSize: '12px',
                  cursor: 'pointer',
                }}
              >
                Done
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
