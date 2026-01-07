import React, { useState } from 'react';
import { questions } from '../questions';
import './Quiz.css';

interface QuizProps {
  onSubmit: (score: number) => void;
}

const Quiz: React.FC<QuizProps> = ({ onSubmit }) => {
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [score, setScore] = useState(0);

  const handleAnswer = (value: number) => {
    setScore(score + value);
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    } else {
      onSubmit(score + value);
    }
  };

  const currentQuestion = questions[currentQuestionIndex];
  const progress = ((currentQuestionIndex) / questions.length) * 100;

  return (
    <div className="quiz-container">
      <div className="progress-bar">
        <div className="progress" style={{ width: `${progress}%` }}></div>
      </div>
      <h2>{currentQuestion.text}</h2>
      <div className="options-container">
        {currentQuestion.options.map((option) => (
          <button key={option.text} onClick={() => handleAnswer(option.value)} className="option-button">
            {option.text}
          </button>
        ))}
      </div>
    </div>
  );
};

export default Quiz;
