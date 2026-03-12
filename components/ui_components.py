import streamlit as st

def render_hero():
    st.markdown("""
    <div class="hero-container animate-fade-in">
        <h1 style='color: white; font-weight: 800; font-size: 3.5rem; margin-bottom: 0;'>Agriconnect 🌱</h1>
        <p style='color: rgba(255,255,255,0.9); font-size: 1.2rem; font-weight: 400;'>Advanced AI Diagnostics for Modern Farming</p>
        <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
            <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 50px; font-size: 0.9rem;">🍃 Pomegranate</div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 50px; font-size: 0.9rem;">🥭 Mango</div>
            <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 50px; font-size: 0.9rem;">🤖 AI-Powered</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_overview():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="prediction-card animate-fade-in">
            <h3 style="margin-top:0;" class="text-main">🔍 Smart Vision</h3>
            <p style="font-size: 0.9rem;" class="text-sub">Upload a photo of your crop, and our dual AI models will analyze it in milliseconds. We use MobileNet and EfficientNet for the best balance of speed and accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="prediction-card animate-fade-in" style="animation-delay: 0.2s;">
            <h3 style="margin-top:0;" class="text-main">🛡️ Early Warning</h3>
            <p style="font-size: 0.9rem;" class="text-sub">Identify 12 unique classes, including high-risk diseases like Bacterial Blight and Anthracnose, before they spread to your entire field.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="prediction-card animate-fade-in" style="animation-delay: 0.4s;">
            <h3 style="margin-top:0;" class="text-main">💬 Expert Chat</h3>
            <p style="font-size: 0.9rem;" class="text-sub">Not sure what to do next? Talk to our AI Agricultural Assistant for immediate treatment recommendations and management tips.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.write("---")

def render_details(data):
    if not data: 
        st.warning("No details found for this class in database.")
        return
    
    category = data.get('Category', 'Unknown')
    title = data.get('Stage_Name') or data.get('Disease_Name') or data.get('Variety_Name')
    
    st.markdown(f"""
    <div class="animate-fade-in" style="margin-top: 2rem;">
        <h2 style='margin-bottom: 0;' class="text-main">{title}</h2>
        <span style='background: #e2e8f0; color: #475569; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;'>{category}</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown(f"""
        <div class="prediction-card" style="margin-top: 1rem; border-left: 5px solid #10b981;">
            <h4 style="margin-top:0;" class="text-main">📖 Description</h4>
            <p style="line-height: 1.6;" class="text-body">{data.get('Description', 'No description available.')}</p>
        </div>
        """, unsafe_allow_html=True)
        
    c1, c2 = st.columns(2)
    with c1:
        if 'Symptoms' in data:
            st.markdown(f"""
            <div class="info-box">
                <h4 style="margin-top:0;" class="text-accent-blue">⚠️ Symptoms</h4>
                <p style="margin-bottom:0;" class="text-main">{data['Symptoms']}</p>
            </div>
            """, unsafe_allow_html=True)
        if 'Characteristics' in data:
            st.markdown(f"""
            <div class="info-box" style="background: rgba(240, 253, 244, 0.8); border-color: #22c55e;">
                <h4 style="margin-top:0;" class="text-accent-green">✨ Characteristics</h4>
                <p style="margin-bottom:0;" class="text-main">{data['Characteristics']}</p>
            </div>
            """, unsafe_allow_html=True)
        if 'Usage' in data:
            with st.expander("🍽️ Usage"):
                st.write(data['Usage'])
                
    with c2:
        if 'Treatment' in data:
             st.markdown(f"""
            <div class="info-box" style="background: rgba(255, 247, 237, 0.8); border-color: #f97316;">
                <h4 style="margin-top:0;" class="text-accent-orange">💊 Treatment</h4>
                <p style="margin-bottom:0;" class="text-main">{data['Treatment']}</p>
            </div>
            """, unsafe_allow_html=True)
        if 'Growing_Conditions' in data:
                st.markdown(f"""
                <div class="info-box" style="background: rgba(250, 245, 255, 0.8); border-color: #a855f7;">
                    <h4 style="margin-top:0;" class="text-accent-purple">☀️ Growing Conditions</h4>
                    <p style="margin-bottom:0;" class="text-main">{data['Growing_Conditions']}</p>
                </div>
                """, unsafe_allow_html=True)

    if 'MANAGEMENT_TIPS' in data:
        st.warning(f"**💡 Management Tips:** {data['MANAGEMENT_TIPS']}")
        
    if 'Images' in data and isinstance(data['Images'], list):
        with st.expander("🖼️ Reference Images", expanded=True):
            cols = st.columns(min(len(data['Images']), 4))
            for idx, img_url in enumerate(data['Images'][:4]): 
                with cols[idx]:
                    st.image(img_url, use_container_width=True)

def render_prediction_cards(name_keras, conf_keras, time_keras, name_pytorch, conf_pytorch, time_pytorch):
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown(f"""
        <div class="prediction-card" style="text-align: center; border-bottom: 5px solid #3b82f6;">
            <h3 style="margin: 0; font-size: 1.2rem;" class="text-accent-blue">MobileNet V2</h3>  
            <p style="font-size: 0.8rem; margin-bottom: 10px;" class="text-sub">Speed Optimized</p>
            <hr style="opacity: 0.1; margin: 15px 0;">
            <h2 style="font-size: 1.8rem; margin: 10px 0;" class="text-main">{name_keras}</h2>
            <div style="background: #e2e8f0; border-radius: 10px; height: 10px; margin: 15px 0; overflow: hidden;">
                <div style="background: #3b82f6; width: {conf_keras*100}%; height: 100%; border-radius: 10px;"></div>
            </div>
            <p style="font-weight: 600;" class="text-accent-blue">{conf_keras:.1%} Confidence</p>
            <p style="font-size: 0.7rem;" class="text-sub">Latency: {time_keras:.1f} ms</p>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown(f"""
        <div class="prediction-card" style="text-align: center; border-bottom: 5px solid #10b981;">
            <h3 style="margin: 0; font-size: 1.2rem;" class="text-accent-green">EfficientNet B4</h3>  
            <p style="font-size: 0.8rem;" class="text-sub">Accuracy Optimized</p>
            <hr style="opacity: 0.1; margin: 15px 0;">
            <h2 style="font-size: 1.8rem; margin: 10px 0;" class="text-main">{name_pytorch}</h2>
            <div style="background: #e2e8f0; border-radius: 10px; height: 10px; margin: 15px 0; overflow: hidden;">
                <div style="background: #10b981; width: {conf_pytorch*100}%; height: 100%; border-radius: 10px;"></div>
            </div>
            <p style="font-weight: 600;" class="text-accent-green">{conf_pytorch:.1%} Confidence</p>
            <p style="font-size: 0.7rem;" class="text-sub">Latency: {time_pytorch:.1f} ms</p>
        </div>
        """, unsafe_allow_html=True)
