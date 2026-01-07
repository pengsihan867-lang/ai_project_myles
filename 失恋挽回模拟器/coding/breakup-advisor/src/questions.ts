export interface Question {
  text: string;
  options: {
    text: string;
    value: number;
  }[];
}

export const questions: Question[] = [
  {
    text: '分手多久了？',
    options: [
      { text: '一周以内', value: 10 },
      { text: '一个月以内', value: 7 },
      { text: '三个月以内', value: 3 },
      { text: '超过三个月', value: 1 },
    ],
  },
  {
    text: '是谁提出的分手？',
    options: [
      { text: '对方提出的', value: 7 },
      { text: '我提出的', value: 3 },
      { text: '共同决定', value: 5 },
      { text: '情况复杂', value: 2 },
    ],
  },
  {
    text: '对方目前有新伴侣吗？',
    options: [
      { text: '没有', value: 10 },
      { text: '不确定', value: 5 },
      { text: '有', value: 0 },
    ],
  },
  {
    text: '你们是否还保持联系？',
    options: [
      { text: '是，像朋友一样', value: 8 },
      { text: '是，但很尴尬或很少', value: 4 },
      { text: '完全没有联系', value: 2 },
    ],
  },
  {
    text: '你认为分手的主要原因是什么？',
    options: [
      { text: '外部因素 (家庭、距离)', value: 8 },
      { text: '误会或沟通不畅', value: 6 },
      { text: '我犯了错', value: 4 },
      { text: '对方的原则性问题', value: 1 },
      { text: '感觉淡了，没有特别原因', value: 2 },
    ],
  },
];
