import React, { useState } from 'react';

interface PaymentGateProps {

}

const PaymentGate: React.FC<PaymentGateProps> = () => {
    const [loading, setLoading] = useState(false);

    const handlePaymentClick = () => {
        setLoading(true);
        // Simulate payment process or redirect to Stripe
        setTimeout(() => {
            // In a real app, this would be a redirect. 
            // For MVP without backend, we simulate a successful return.
            // We will refresh the page with ?paid=true to simulate the return from Stripe.
            window.location.search = '?paid=true';
        }, 1500);
    };

    return (
        <div className="payment-container">
            <div className="locked-preview">
                <h2>Analysis Computing...</h2>
                <div className="blur-content">
                    <p>Based on your answers, we have identified a significant pattern in your relationship dynamic.</p>
                    <p>Your attachment style indicates a strong tendency towards...</p>
                    <p>The breakup was likely triggered by mismatch in...</p>
                </div>
            </div>

            <div className="gate-overlay">
                <h2>Your Assessment is Ready <br /><span className="subtitle-cn">分析报告已生成</span></h2>
                <p>
                    We've analyzed your responses. Your situation shows a distinct pattern that is common but often misunderstood.
                    Unlock the full report to understand the dynamics at play.
                    <br />
                    <span className="subtitle-cn">
                        我们已根据您的回答完成分析。您的情感状况呈现出一种典型但常被忽视的模式。解锁完整报告，看清关系背后的真相。
                    </span>
                </p>

                <button className="unlock-btn" onClick={handlePaymentClick} disabled={loading}>
                    {loading ? 'Processing...' : 'Unlock Full Report for $2.99 (¥9.90)'}
                </button>

                <p className="secure-note">
                    <span role="img" aria-label="lock">🔒</span> Secure payment via Stripe. One-time fee. No subscription.
                </p>
            </div>
        </div>
    );
};

export default PaymentGate;
