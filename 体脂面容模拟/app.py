import streamlit as st
from PIL import Image
import utils

# 设置页面配置
st.set_page_config(
    page_title="AI 体脂重塑 (AI Body Fat Transformer)",
    page_icon="💪",
    layout="wide"
)

# 侧边栏配置
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

with st.sidebar:
    st.header("⚙️ 设置")
    # 尝试从环境变量获取默认 Key
    default_key = os.getenv("ZHIPUAI_API_KEY", "")
    api_key = st.text_input("请输入智谱 API Key", value=default_key, type="password", help="请前往智谱 AI 开放平台获取")
    if not api_key:
        st.warning("⚠️ 请输入 API Key 以继续使用功能")
        st.stop() # 停止后续代码执行

# 初始化智谱客户端
try:
    client = utils.get_zhipu_client(api_key)
except Exception as e:
    st.error(f"客户端初始化失败: {e}")
    st.stop()

# 主界面标题
st.title("💪 AI 体脂重塑 (AI Body Fat Transformer)")
st.markdown("上传照片 -> AI 分析体脂 -> 设定目标 -> 生成新形象")

# 步骤一：上传与分析
st.header("1. 上传照片")
uploaded_file = st.file_uploader("请上传一张原本的人像照片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显式原图
    image = Image.open(uploaded_file)
    
    # 布局：左侧原图，右侧信息
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="原始照片", use_container_width=True)

    # 自动触发分析
    # 使用 session_state 防止重复分析
    if "analyzed_body_fat" not in st.session_state or st.session_state.uploaded_file_id != uploaded_file.file_id:
        with st.spinner("正在分析体脂率..."):
            try:
                # 记录当前文件ID，避免重复分析
                st.session_state.uploaded_file_id = uploaded_file.file_id
                
                # 调用 utils 进行分析
                body_fat = utils.analyze_body_fat(client, image)
                st.session_state.analyzed_body_fat = body_fat
                
                # 重置生成结果
                if "generated_image_url" in st.session_state:
                    del st.session_state.generated_image_url
                    
            except Exception as e:
                st.error(f"分析失败: {e}")
                st.stop()
    
    # 显示分析结果
    current_body_fat = st.session_state.analyzed_body_fat
    with col2:
        st.success("分析完成！")
        st.metric(label="当前估算体脂率", value=f"{current_body_fat}%")

    # 步骤二：设定目标
    st.header("2. 设定目标")
    target_body_fat = st.slider(
        "选择目标体脂率",
        min_value=5.0,
        max_value=40.0,
        value=float(current_body_fat), # 默认值为当前体脂率
        step=1.0,
        format="%f%%"
    )
    
    # 计算差值
    diff = target_body_fat - current_body_fat
    if diff < 0:
        st.info(f"目标：减脂 {abs(diff):.1f}%")
    elif diff > 0:
        st.info(f"目标：增脂/增肌 {diff:.1f}%")
    else:
        st.info("目标：保持当前体脂率")

    # 步骤三：生成新形象
    st.header("3. 生成新形象")
    
    if st.button("✨ 生成预测照", type="primary"):
        with st.spinner("AI 正在重塑形象，请稍候... (可能需要几秒到十几秒)"):
            try:
                # 调用 utils 生成图片
                # 为了达到最佳效果，这里简单调用。进阶版应该先用 GLM-4V 生成详细 Image Caption
                # 但根据需求 backlog，我们直接调用 cogview-3-plus
                
                # 提示用户：由于是 Demo，效果主要依赖 Prompt 对抗，可能不如专业的 StyleGAN 稳定
                
                new_image_url = utils.generate_new_look(
                    client,
                    current_body_fat,
                    target_body_fat,
                    image
                )
                st.session_state.generated_image_url = new_image_url
                
            except Exception as e:
                st.error(f"生成失败: {e}")

    # 显示生成结果对比
    if "generated_image_url" in st.session_state:
        st.markdown("---")
        st.subheader("对比效果")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.image(image, caption=f"原图 (体脂 {current_body_fat}%)", use_container_width=True)
        with res_col2:
            st.image(st.session_state.generated_image_url, caption=f"预测效果 (体脂 {target_body_fat}%)", use_container_width=True)
