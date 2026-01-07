import { useState, useEffect } from 'react';
import Welcome from './components/Welcome';
import Quiz from './components/Quiz';
import Result from './components/Result';
import LandingPage from './components/LandingPage';
import PaymentGate from './components/PaymentGate';
import './App.css';

type AppState = 'landing' | 'welcome' | 'quiz' | 'locked_result' | 'payment_gate' | 'full_result';

function App() {
  const [appState, setAppState] = useState<AppState>('landing');
  const [score, setScore] = useState(0);

  useEffect(() => {
    // Check for payment success parameter
    const params = new URLSearchParams(window.location.search);
    if (params.get('paid') === 'true') {
      // In a real app we would verify this token with backend
      // Ideally we should persist the score in localStorage so it survives the redirect
      const savedScore = localStorage.getItem('breakup_advisor_score');
      if (savedScore) {
        setScore(parseInt(savedScore, 10));
        setAppState('full_result');
      } else {
        // Fallback if no score found (maybe direct link access), go to landing
        setAppState('landing');
      }
    }
  }, []);

  const handleStartLanding = () => {
    setAppState('welcome');
  };

  const handleStartQuiz = () => {
    setAppState('quiz');
  };

  const handleSubmit = (finalScore: number) => {
    setScore(finalScore);
    localStorage.setItem('breakup_advisor_score', finalScore.toString());
    setAppState('locked_result');
  };



  const renderContent = () => {
    switch (appState) {
      case 'landing':
        return <LandingPage onStart={handleStartLanding} />;
      case 'welcome':
        return <Welcome onStart={handleStartQuiz} />;
      case 'quiz':
        return <Quiz onSubmit={handleSubmit} />;
      case 'locked_result':
      case 'payment_gate':
        return <PaymentGate />; // We combine these visual states for simplicity in MVP or separate if needed. 
      // Actually, let's make it simple: 'locked_result' shows the blurred preview + button calling 'payment_gate' logic.
      // In the component I made, PaymentGate handles both visual aspects.
      case 'full_result':
        return <Result score={score} />;
      default:
        return <LandingPage onStart={handleStartLanding} />;
    }
  };

  return (
    <div className="container">
      {renderContent()}
    </div>
  )
}

export default App;
