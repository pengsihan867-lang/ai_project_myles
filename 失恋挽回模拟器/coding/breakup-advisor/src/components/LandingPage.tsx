import React from 'react';

interface LandingPageProps {
    onStart: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onStart }) => {
    return (
        <div className="landing-container">
            <h1>Breakup Clarity Assessment <br /><span className="subtitle-cn">失恋迷雾分析</span></h1>
            <p className="subtitle">
                Navigating the end of a relationship is hard. Get an objective perspective on what happened and where to go next.
                <br />
                <span className="subtitle-cn">分手并不意味着失败。客观理清现状，找回内心的平静。</span>
            </p>

            <div className="alert-box">
                <p><strong>Private & Secure</strong>: No accounts. No data selling. Just answers.</p>
            </div>

            <button className="primary-btn" onClick={onStart}>
                Start Assessment <span className="btn-cn">开始分析</span>
            </button>

            <p className="disclaimer">
                Disclaimer: This tool is for self-reflection only. Not a substitute for professional therapy.
            </p>
        </div>
    );
};

export default LandingPage;
