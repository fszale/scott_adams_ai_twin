__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import requests
import chromadb
from sentence_transformers import SentenceTransformer
import os

# Initialize Grok API
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
GROK_API_KEY = st.secrets["grok_api_key"]  # Set in Streamlit secrets

# Initialize Chroma and Sentence-BERT
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("scott_adams_insights")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load knowledge base
def load_knowledge_base():
    documents = []
    for file in os.listdir("./knowledge_base"):
        with open(f"./knowledge_base/{file}", "r") as f:
            documents.append(f.read())
    embeddings = embedder.encode(documents)
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        collection.add(ids=[str(i)], embeddings=[emb.tolist()], documents=[doc])

# Query Grok API
def query_grok(prompt, context):
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    system_prompt = f"""
    You are an AI persona based on Scott Adams, creator of *Dilbert* and host of *Coffee With Scott Adams*, embodying the full scope of his written works, including *How to Fail at Almost Everything and Still Win Big*, *Win Bigly*, *The Dilbert Principle*, *The Joy of Work*, *The Dilbert Future*, *God's Debris*, *Loserthink*, *Creativity*, and others. Your tone is witty, satirical, and contrarian, like Scott sipping coffee while skewering nonsense. Use the provided knowledge base: {context}.

    Core perspectives:
    - **Systems over Goals** (*How to Fail*): Success comes from building systems (e.g., daily habits, talent stacking) to increase odds of luck, not chasing specific goals. Failure is a tool—embrace it to learn and pivot.
    - **Persuasion Mastery** (*Win Bigly*): Humans are 'moist robots,' swayed by emotion, not logic. Use techniques like pacing and leading, visual imagery, and high-ground maneuvers to predict and influence behavior.
    - **Talent Stacking** (*How to Fail*): Combine average skills (e.g., public speaking, psychology, humor) into a unique, unbeatable stack.
    - **Creativity Frameworks** (*Creativity*): Creativity is a system—combine old ideas, break rules, and iterate fast. Failure fuels innovation.
    - **Corporate Absurdity** (*The Dilbert Principle*, *The Joy of Work*): Bureaucracy is a cartoonish mess—outsmart it with skepticism, humor, and minimal meetings.
    - **Future Prediction** (*The Dilbert Future*, *God's Debris*): Spot trends by tracking incentives, human nature, and technology. Question 'settled' narratives.
    - **Propaganda Busting** (*Loserthink*, *Win Bigly*): Decode corporate and government spin by ignoring headlines, checking primary sources (e.g., X posts), and reasoning from first principles.
    - **Reasoning and Skepticism** (*Loserthink*): Avoid 'loserthink'—lazy mental habits like mind-reading or overgeneralizing. Use clear, probabilistic thinking.

    Respond with sharp, humorous insights on predicting the future, reasoning clearly, navigating politics, or dismantling corporate and media propaganda. Tackle business, creativity, politics, or life strategies with no topic off-limits. If relevant, weave in persuasion tricks, talent stacking, or systems thinking. If you don't know the answer, say 'I don't know, but life's a cartoon—draw your own conclusion.' Use the knowledge base for accuracy, falling back on general knowledge if needed. Keep responses engaging, never dull, and always sound like Scott chuckling at the absurdity of it all.
    """
    data = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }
    response = requests.post(GROK_API_URL, json=data, headers=headers)
    try:
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        elif "error" in result:
            return f"Error: {result['error']}"
        else:
            return f"Unexpected response: {result}"
    except Exception as e:
        return f"Failed to parse response: {e}"

# Streamlit app
# Health endpoint
if st.query_params.get("path", "") == "health":
    st.write({"status": "healthy", "message": "App is running"})
    st.stop()

# Add custom CSS for background image
st.markdown("""
<style>
    .stApp {
        background-image: url("https://raw.githubusercontent.com/fszale/scott_adams_ai_twin/main/scott_adams.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh; /* Ensure app takes at least full viewport height */
        overflow-y: auto; /* Explicitly allow vertical scrolling */
        opacity: 1; /* Keep app container fully opaque */
        position: relative; /* Positioning context for pseudo-element */
    }
    .stApp::before {
        content: "";
        position: fixed; /* Use fixed to cover viewport without affecting scroll */
        top: 0;
        left: 0;
        width: 100%;
        height: 100vh; /* Cover full viewport height */
        background-image: inherit;
        background-size: inherit;
        background-position: inherit;
        background-repeat: inherit;
        opacity: 0.05; /* Very low opacity for barely visible background */
        z-index: -1; /* Place behind all content */
    }
    .stApp > div {
        background: rgba(255, 255, 255, 0.95); /* Semi-transparent white overlay for readability */
        padding: 20px;
        border-radius: 10px;
        position: relative; /* Ensure content stays above background */
        z-index: 1; /* Place content above pseudo-element */
        min-height: 100%; /* Allow content to expand */
        box-sizing: border-box; /* Ensure padding doesn't cause overflow */
    }
    /* Ensure chat input stays at the bottom and is visible */
    .stChatInput {
        position: sticky;
        bottom: 0;
        z-index: 2; /* Above content and background */
        background: rgba(255, 255, 255, 1); /* Solid background for input */
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Coffee With Scott Adams' AI Twin")

# UI introduction
st.markdown("---")
st.markdown("""
#### Sip Coffee with Scott Adams (Virtually)!
I'm the AI version of Scott Adams, *Dilbert* creator and master of cutting through BS. From *Coffee With Scott Adams* and books like *How to Fail at Almost Everything*, *Win Bigly*, and *Loserthink*, I teach you to predict the future, reason like a pro, and spot corporate and government propaganda from a mile away. Ask about business, politics, creativity, or why the news is lying to you. Let’s dismantle some nonsense together!
""")
st.markdown("---")
st.markdown("Ask about predicting trends, outsmarting bureaucracy, or decoding media spin. Go ahead, make my coffee spill!")

# Load knowledge base on first run
if not collection.count():
    load_knowledge_base()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("What's the latest corporate or media lie you've spotted?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve relevant documents
    query_embedding = embedder.encode([prompt])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
    context = " ".join(results["documents"][0])
    response = query_grok(prompt, context)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})