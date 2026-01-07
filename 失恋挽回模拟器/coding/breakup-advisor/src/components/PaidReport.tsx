import React from 'react';
import './PaidReport.css';

const PaidReport: React.FC = () => {
  return (
    <div className="paid-report-container">
      <h2>初步行动方向报告</h2>

      <div className="report-section">
        <h3>第一部分：自我调整建议 (心态建设)</h3>
        <p><strong>1. 接受现实，管理情绪：</strong> 承认分手的现实，允许自己有悲伤和失落的情绪，但不要沉溺其中。尝试通过运动、冥想、与朋友倾诉等方式来疏导情绪。</p>
        <p><strong>2. 降低需求感，重建个人生活：</strong> 停止一切高需求的纠缠行为。把注意力放回自己身上，重新拾起你的兴趣爱好，结交新朋友，提升自己的价值。你越是精彩，吸引力就越强。</p>
        <p><strong>3. 客观复盘，而非自我指责：</strong> 理性分析关系中的问题，思考双方的责任，但避免过度的自我攻击。这次经历是你成长的机会。</p>
      </div>

      <div className="report-section">
        <h3>第二部分：复盘关系常见误区</h3>
        <p><strong>1. 沟通不畅：</strong> 是否存在有效沟通的障碍？是一方总在回避，还是双方都倾向于指责？</p>
        <p><strong>2. 价值失衡：</strong> 在关系中，你是否过度付出，失去了自己的框架和吸引力？</p>
        <p><strong>3. 忽视核心需求：</strong> 你是否真正了解对方的核心情感需求？（例如：是需要陪伴、理解，还是需要支持和崇拜？）</p>
      </div>

      <div className="report-section">
        <h3>第三部分：潜在接触时机与原则</h3>
        <p><strong>1. "断联"并非必须，"冷处理"是关键：</strong> 如果你们还能做普通朋友，不必刻意删除所有联系方式。关键是降低联系频率，内容以无需求的、积极的分享为主。</p>
        <p><strong>2. 最佳复联时机：</strong> 通常在你完成自我调整，朋友圈等社交展示面也焕然一新之后。当对方开始对你的改变产生好奇时，就是最好的时机。</p>
        <p><strong>3. "测试性"接触：</strong> 可以通过共同朋友的聚会、点赞对方不涉及情感的朋友圈等方式，进行低成本的测试。观察对方的反应，如果反应积极，再考虑下一步。</p>
      </div>

      <div className="report-disclaimer">
        <p><strong>免责声明：</strong> 本报告提供的是基于通用心理学原则的初步方向指导，不能替代专业的心理咨询。每个人的情感状况都是独特的，请结合自身情况，理性决策。</p>
      </div>
    </div>
  );
};

export default PaidReport;