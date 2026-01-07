import React from 'react';

interface ResultProps {
  score: number;
}

const getResultContent = (score: number) => {
  // Simple logic to map score to patterns
  let pattern = "";
  let emotion = "";
  let perspective = "";

  if (score < 15) {
    pattern = "Anxious-Avoidant Dynamic (焦虑-回避型动力)";
    emotion = "confusion and a sense of abandonment (困惑与被抛弃感)";
    perspective = "the need for immediate space rather than forced closeness (需要空间而非强行靠近)";
  } else if (score <= 30) {
    pattern = "Communication Breakdown (沟通断裂)";
    emotion = "frustration and misunderstanding (挫败感与误解)";
    perspective = "a temporary misalignment that needs patience (需要耐心的暂时性错位)";
  } else {
    pattern = "Situational Strain (情境性压力)";
    emotion = "regret and hope (遗憾与希望)";
    perspective = "external factors overshadowing emotional bond (外部因素掩盖了情感联结)";
  }

  return { pattern, emotion, perspective };
};

const Result: React.FC<ResultProps> = ({ score }) => {
  const { pattern, emotion, perspective } = getResultContent(score);

  return (
    <div className="result-container">
      <div className="result-header">
        <h1>Assessment Complete <br /><span className="subtitle-cn">分析完成</span></h1>
        <div className="score-circle">
          <span>{score}</span>
          <small>/ 50</small>
        </div>
      </div>

      <div className="result-section">
        <h3>1. Pattern Identification <span className="subtitle-cn">模式识别</span></h3>
        <p>
          Based on your inputs, your relationship ended primarily due to factors aligning with the <strong>{pattern}</strong>.
          <br />
          <span className="subtitle-cn">根据您的输入，这段关系的结束主要是由于符合【{pattern}】的因素。</span>
        </p>
      </div>

      <div className="result-section">
        <h3>2. Emotional Context <span className="subtitle-cn">情感与其背景</span></h3>
        <p>
          In this situation, it is normal to feel <strong>{emotion}</strong>. Connection requires two aligned people, and currently, the alignment is broken.
          <br />
          <span className="subtitle-cn">在这种情况下，感到【{emotion}】是正常的。连接需要两个人的同频，而目前这种同频已被打破。</span>
        </p>
      </div>

      <div className="result-section">
        <h3>3. Perspective Shift <span className="subtitle-cn">视角转换</span></h3>
        <p>
          While the outcome hurts, it often signals <strong>{perspective}</strong>. Situations like this can change, but currently, acceptance is the most powerful tool.
          <br />
          <span className="subtitle-cn">虽然结果令人痛苦，但这通常预示着【{perspective}】。局面可能会改变，但目前，“接受”是最有力量的工具。</span>
        </p>
      </div>

      <div className="result-section">
        <h3>Next Steps <span className="subtitle-cn">下一步</span></h3>
        <ul className="todo-list">
          <li><strong>No Contact Rule (30 Days)</strong>: Give emotions time to settle. <br /><span className="subtitle-cn">断联原则（30天）：让情绪沉淀。</span></li>
          <li><strong>Focus on Self (Self-Care)</strong>: Rebuild your own identity. <br /><span className="subtitle-cn">关注自我：重建你的个人身份感。</span></li>
        </ul>
      </div>

      <button className="restart-btn" onClick={() => window.location.href = '/'}>
        Start New Assessment
      </button>

      <p className="disclaimer">
        This is a generated report based on your inputs.
      </p>
    </div>
  );
};

export default Result;
