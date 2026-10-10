import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';
import { BackendProvider } from './context/BackendContext';

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <BackendProvider>
      <App />
    </BackendProvider>
  </React.StrictMode>,
);
