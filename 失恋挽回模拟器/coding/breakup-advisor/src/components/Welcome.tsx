import React from 'react';
import './Welcome.css';

interface WelcomeProps {
  onStart: () => void;
}

const Welcome: React.FC<WelcomeProps> = ({ onStart }) => {
  return (
    <div className="welcome-container">
      <h1>评估你的挽回机会</h1>
      <p>这个快速测试将帮助你了解在当前情况下挽回前任的可能性。</p>
      <button onClick={onStart} className="start-button">开始测试</button>
    </div>
  );
};

export default Welcome;
