import streamlit as st
import requests

# Set up the page layout
st.set_page_config(page_title="India AI Fact Checker", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ India Multi-Agent Fact Checker")
st.markdown("Paste a news article or WhatsApp forward below. Our **BiLSTM** and **LangGraph Agents** will verify it in real-time.")

# The text box for user input
news_text = st.text_area("News Text:", height=150, placeholder="Enter a claim here (e.g., NASA found water on Mars)...")

# The Verify Button
if st.button("Verify News", type="primary"):
    if not news_text.strip():
        st.warning("Please enter some text.")
    else:
        # Show a loading spinner while LangGraph runs
        with st.spinner("🤖 AI Agents are investigating... (This takes 10-20 seconds)"):
            try:
                # Send the text to your FastAPI main.py server
                response = requests.post("http://localhost:8000/verify", json={"text": news_text})
                
                # If LangGraph succeeded (The 200 OK you saw in your logs)
                if response.status_code == 200:
                    data = response.json()
                    verdict = data['agent_verdict']['overall_verdict']
                    score = data['agent_verdict']['truth_score']
                    
                    # Color coding the verdict
                    color = "green" if verdict == "TRUE" else "red" if verdict == "FALSE" else "orange"
                    
                    # --- TOP LEVEL VERDICT ---
                    st.markdown(f"### 🎯 Final Verdict: :{color}[**{verdict}**]")
                    st.progress(score / 100)
                    
                    st.divider()
                    
                    # --- SPLIT INTO TWO COLUMNS ---
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🧠 BiLSTM Structural Check")
                        st.info("This AI checks if the text is written like typical Fake News.")
                        bilstm = data['bilstm_structural_analysis']
                        if "error" in bilstm:
                            st.error(f"Model Error: {bilstm['error']}")
                        else:
                            st.metric(label="Pattern Detected", value=bilstm['structural_label'])
                            st.caption(f"Confidence: {bilstm['structural_confidence']:.2%}")
                            
                    with col2:
                        st.subheader("🔎 LangGraph Research Agent")
                        st.info("This Agent searched Google News, DuckDuckGo, and Wikipedia.")
                        st.metric(label="Truth Score", value=f"{score}/100")
                        
                    # --- CLAIM BY CLAIM BREAKDOWN ---
                    st.divider()
                    st.subheader("📋 Claim-by-Claim Breakdown")
                    for i, claim in enumerate(data['detailed_claims']):
                        emoji = "✅" if claim['verdict'] == "TRUE" else "❌" if claim['verdict'] == "FALSE" else "⚠️"
                        with st.expander(f"{emoji} Claim {i+1}: {claim['verdict']}"):
                            st.write(f"**Claim:** {claim['claim']}")
                            st.write(f"**Gemini's Explanation:** {claim['explanation']}")
                            
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to Backend! Make sure `python main.py` is running in another terminal.")