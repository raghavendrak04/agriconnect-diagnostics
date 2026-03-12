import google.generativeai as genai
import streamlit as st

def configure_gemini(api_key):
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return False

def get_chatbot_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        context_prompt = """
        You are an expert Agricultural Consultant for 'Agriconnect', a specialized tool for Pomegranate and Mango farmers.
        
        Your Knowledge Base:
        1. Crops: Pomegranate (Bhagwa, Arakta) and Mango (Kesar, Alphonso, Dasheri).
        2. Diseases: Alternaria, Anthracnose, Bacterial Blight, Cercospora, Powdery Mildew.
        3. Pests: Thrips, Fruit Borer, Stem Borer.
        4. Treatments: Recommend both Organic (Neem oil, etc.) and Chemical (Fungicides like Mancozeb) remedies with dosage if possible.
        
        Guidelines:
        - Be concise and practical. Farmers need actionable advice.
        - If the user greets you, welcome them to Agriconnect.
        - If the question is NOT about agriculture, answer briefly but remind them you are an Agri-Expert.
        - Use formatting (bullet points) for readability.
        """
        full_prompt = f"{context_prompt}\n\nUser Question: {prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

def render_chatbot():
    # --- Header Row (Title + Clear Button) ---
    # We use columns to allow the clear button to be a native Streamlit button
    with st.container():
        h_col1, h_col2 = st.columns([5, 1], gap="small")
        with h_col1:
            st.markdown('<div class="chat-title"><span style="font-size: 1.2rem;">🤖</span> Agriconnect Assistant</div>', unsafe_allow_html=True)
        with h_col2:
            if st.button("🗑️", key="clear_chat_btn", help="Clear Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Chat History
    chat_container = st.container(height=400) # Increased height slightly
    with chat_container:
        if len(st.session_state.messages) == 0:
            # Empty State: Welcome Message + Suggestions
            st.markdown("""
            <div style="padding: 20px; text-align: center;" class="text-sub">
                <p>👋 <strong>Welcome!</strong> I'm here to help with your Pomegranate and Mango farming needs.</p>
                <p style="font-size: 0.85rem;">Ask about diseases, treatments, or growth stages.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='padding: 0 15px 5px 15px; font-weight: 600; font-size: 0.9rem;' class='text-sub'>Suggested Topics</div>", unsafe_allow_html=True)
            suggestions = [
                "🛡️ How to treat Bacterial Blight?",
                "🥭 Best Mango varieties for Gujarat?",
                "💊 Organic remedies for Anthracnose",
                "🍃 Pomegranate growth stages"
            ]
            
            # Render suggestions as buttons
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=True):
                    # Process suggestion as a prompt
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    with st.spinner("Thinking..."):
                        text_response = get_chatbot_response(suggestion)
                        st.session_state.messages.append({"role": "assistant", "content": text_response})
                    st.rerun()
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Type your question...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    text_response = get_chatbot_response(prompt)
                    st.markdown(text_response)
                    st.session_state.messages.append({"role": "assistant", "content": text_response})
