import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

console.log('App mounting...');
const root = document.getElementById('root');
console.log('Root element:', root);

createRoot(root!).render(
  <StrictMode>
    <App />
  </StrictMode>
);

console.log('App mounted');
