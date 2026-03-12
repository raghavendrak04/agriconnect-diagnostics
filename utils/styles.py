import streamlit as st

def apply_custom_css(icon_base64):
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

        /* Main Container Styling */
        .stApp {{
            background: radial-gradient(circle at top right, #f8fafc, #eff6ff);
        }}

        [data-theme="dark"] .stApp {{
            background: radial-gradient(circle at top right, #0f172a, #1e1b4b);
        }}

        html, body, [class*="css"] {{
            font-family: 'Outfit', sans-serif;
        }}

        /* Animations */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        /* --- THEME AWARE TYPOGRAPHY --- */
        .text-main {{ color: #1e293b; }}
        [data-theme="dark"] .text-main {{ color: #f1f5f9 !important; }}

        .text-sub {{ color: #475569; }}
        [data-theme="dark"] .text-sub {{ color: #94a3b8 !important; }}
        
        .text-body {{ color: #334155; }}
        [data-theme="dark"] .text-body {{ color: #cbd5e1 !important; }}

        .text-accent-blue {{ color: #0369a1; }}
        [data-theme="dark"] .text-accent-blue {{ color: #38bdf8 !important; }}

        .text-accent-green {{ color: #15803d; }}
        [data-theme="dark"] .text-accent-green {{ color: #4ade80 !important; }}
        
        .text-accent-orange {{ color: #9a3412; }}
        [data-theme="dark"] .text-accent-orange {{ color: #fb923c !important; }}

        .text-accent-purple {{ color: #7e22ce; }}
        [data-theme="dark"] .text-accent-purple {{ color: #c084fc !important; }}

        @keyframes pulse-subtle {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.03); }}
            100% {{ transform: scale(1); }}
        }}

        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-10px); }}
            100% {{ transform: translateY(0px); }}
        }}

        .animate-fade-in {{ animation: fadeInUp 0.8s ease-out; }}
        .animate-float {{ animation: float 3s ease-in-out infinite; }}

        /* Hero Section */
        .hero-container {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 3rem 2rem;
            border-radius: 24px;
            color: white;
            text-align: center;
            margin-bottom: 2.5rem;
            box-shadow: 0 20px 40px rgba(16, 185, 129, 0.2);
            position: relative;
            overflow: hidden;
        }}

        .hero-container::before {{
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
            animation: pulse-subtle 4s infinite;
        }}

        /* Cards & Components */
        .prediction-card {{
            background: rgba(255, 255, 255, 0.8) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            padding: 24px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        }}

        [data-theme="dark"] .prediction-card {{
            background: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .prediction-card:hover {{
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 50px rgba(0,0,0,0.1);
        }}

        .stButton>button {{
            background: linear-gradient(90deg, #10b981, #3b82f6);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 12px;
            font-weight: 600;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 8px 20px rgba(59, 130, 246, 0.4);
            background: linear-gradient(90deg, #059669, #2563eb);
        }}

        /* Info Boxes */
        .info-box {{
            background: #f0f9ff;
            border-left: 5px solid #0ea5e9;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
        }}

        [data-theme="dark"] .info-box {{
            background: #0c4a6e;
            color: #e0f2fe;
        }}

        /* SideBar Styling */
        [data-testid="stSidebar"] {{
            background-color: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }}

        [data-theme="dark"] [data-testid="stSidebar"] {{
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }}

        /* Floating Action Button (FAB) Styling */
        div[data-testid="stPopover"] {{
            position: fixed !important;
            bottom: 40px !important;
            right: 40px !important;
            z-index: 99999 !important;
        }}
        
        @keyframes pulse-ring {{
            0% {{ transform: scale(.33); opacity: 1; }}
            80%, 100% {{ opacity: 0; }}
        }}

        div[data-testid="stPopover"]::before {{
            content: '';
            position: absolute;
            left: 5px;
            top: 5px;
            width: 55px;
            height: 55px;
            background-color: #0072FF;
            border-radius: 50%;
            animation: pulse-ring 2s cubic-bezier(0.455, 0.03, 0.515, 0.955) infinite;
        }}

        /* Chat Expert Label */
        div[data-testid="stPopover"]::after {{
            content: 'Chat Expert';
            position: absolute;
            right: 75px;
            top: 50%;
            transform: translateY(-50%);
            background: #0072FF;
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            white-space: nowrap;
            box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
            pointer-events: none;
            opacity: 0;
            transition: all 0.3s ease;
        }}

        div[data-testid="stPopover"]:hover::after {{
            opacity: 1;
            right: 85px;
        }}

        /* Button Styling */
        div[data-testid="stPopover"] button {{
            width: 65px !important;
            height: 65px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #00C6FF, #0072FF) !important; 
            border: 3px solid white !important;
            box-shadow: 0 10px 25px rgba(0, 114, 255, 0.4) !important;
            background-image: url("data:image/png;base64,{icon_base64}") !important;
            background-size: 70% !important; 
            background-repeat: no-repeat !important;
            background-position: center !important;
            transition: all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            padding: 0 !important;
        }}

        div[data-testid="stPopover"] button:hover {{
            transform: translateY(-8px) rotate(5deg) !important;
            box-shadow: 0 15px 35px rgba(0, 114, 255, 0.6) !important;
        }}
        
        /* Chatbot Popover Window Styling */
        div[data-testid="stPopoverContent"], div[data-testid="stPopoverBody"] {{
            border-radius: 20px !important;
            padding: 0 !important;
            overflow: hidden !important;
            border: 1px solid rgba(0,0,0,0.1) !important;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2) !important;
            position: fixed !important;
            bottom: 115px !important;
            right: 40px !important;
            left: auto !important;
            transform: none !important;
            min-width: 380px !important;
            z-index: 100000 !important;
        }}

        /* Remove default Streamlit padding inside the popover */
        div[data-testid="stPopoverContent"] [data-testid="stVerticalBlock"],
        div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] {{
            padding: 0 !important;
            gap: 0 !important;
        }}

        div[data-testid="stPopoverContent"] .stColumn,
        div[data-testid="stPopoverBody"] .stColumn {{
            padding: 0 !important;
        }}

        /* --- Chatbot Header & Layout --- */
        
        /* Target the FIRST horizontal block inside the popover to act as the header */
        div[data-testid="stPopoverBody"] > div > div > div[data-testid="stHorizontalBlock"]:first-child {{
            background: linear-gradient(90deg, #00C6FF, #0072FF);
            padding: 10px 15px;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
            margin-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        /* Chatbot Icon/Title in Header */
        .chat-title {{
            color: white;
            font-weight: 600;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        /* Trash/Clear Button Styling to blend into header */
        /* Target only the button in the header (first horizontal block) if possible, or just all secondary buttons in popover */
        div[data-testid="stPopoverBody"] button[kind="secondary"] {{
            background: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            height: 35px;
            padding: 0 10px !important;
            min-height: unset !important;
        }}
        
        div[data-testid="stPopoverBody"] button[kind="secondary"]:hover {{
            background: rgba(255, 255, 255, 0.3) !important;
            border-color: white !important;
            color: white !important;
        }}

        /* Adjust vertical block padding to remove gaps around header */
        div[data-testid="stPopoverBody"] > div {{
            padding-top: 0 !important;
        }}


        /* Improved Chat Message Styling */
        div[data-testid="stChatMessage"] {{
            background: transparent !important;
            padding: 10px 0 !important;
        }}

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {{
            font-size: 0.95rem;
            line-height: 1.5;
        }}
        
        /* User Message Bubble */
        div[data-testid="chatAvatarIcon-user"] + div [data-testid="stMarkdownContainer"] {{
            background: #e0f2fe;
            color: #0f172a;
            padding: 12px 16px;
            border-radius: 16px 16px 0 16px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-left: auto;
            border: 1px solid #bae6fd;
        }}

        [data-theme="dark"] div[data-testid="chatAvatarIcon-user"] + div [data-testid="stMarkdownContainer"] {{
            background: #0369a1;
            color: white;
            border-color: #075985;
        }}

        /* Assistant Message Bubble */
        div[data-testid="chatAvatarIcon-assistant"] + div [data-testid="stMarkdownContainer"] {{
            background: #f1f5f9;
            color: #1e293b;
            padding: 12px 16px;
            border-radius: 16px 16px 16px 0;
            border: 1px solid #e2e8f0;
        }}

        [data-theme="dark"] div[data-testid="chatAvatarIcon-assistant"] + div [data-testid="stMarkdownContainer"] {{
            background: #334155;
            color: #f1f5f9;
            border-color: #475569;
        }}

        .chat-suggestion-container {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 15px;
            background: #f8fafc;
            border-bottom: 1px solid #e2e8f0;
        }}

        .chat-suggestion-chip {{
            background: white;
            border: 1px solid #e2e8f0;
            padding: 12px 16px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
            color: #1e293b;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 500;
            text-align: left;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}

        .chat-suggestion-chip:hover {{
            border-color: #0072FF;
            color: #0072FF;
            background: #f0f7ff;
            transform: translateX(5px);
        }}

        .chat-suggestion-chip span {{
            color: #0072FF;
            font-size: 1.2rem;
            font-weight: bold;
        }}

        .stChatFloatingInputContainer {{
            background: white !important;
            border-top: 1px solid #e2e8f0 !important;
        }}

        /* Chatbot Suggestion Buttons Styling */
        div[data-testid="stPopoverContent"] .stButton>button,
        div[data-testid="stPopoverBody"] .stButton>button {{
            background: white !important;
            color: #1a1a1a !important;
            border: 1px solid #e2e8f0 !important;
            text-align: left !important;
            justify-content: flex-start !important;
            text-transform: none !important;
            letter-spacing: normal !important;
            font-weight: 500 !important;
            padding: 12px 16px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
        }}

        div[data-testid="stPopoverContent"] .stButton>button:hover {{
            border-color: #0072FF !important;
            color: #0072FF !important;
            background: #f0f7ff !important;
            transform: translateX(8px) !important;
            box-shadow: 0 4px 12px rgba(0, 114, 255, 0.1) !important;
        }}
    </style>
    """, unsafe_allow_html=True)
