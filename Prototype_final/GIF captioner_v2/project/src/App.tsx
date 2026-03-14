import { useState } from 'react';
import LandingPage from './components/LandingPage';
import MainApp from './components/MainApp';

function App() {
  const [currentPage, setCurrentPage] = useState<'landing' | 'app'>('landing');

  return (
    <>
      {currentPage === 'landing' ? (
        <LandingPage onGetStarted={() => setCurrentPage('app')} />
      ) : (
        <MainApp onBackToHome={() => setCurrentPage('landing')} />
      )}
    </>
  );
}

export default App;
