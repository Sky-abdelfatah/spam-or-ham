import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np

# =============================================
# Page Config
# =============================================
st.set_page_config(
    page_title="Spam Detector",
    layout="centered"
)

# =============================================
# NLTK Data Download
# =============================================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# =============================================
# Load Model & Vectorizer
# =============================================
@st.cache_resource
def load_model():
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    return vectorizer, model

vectorizer, model = load_model()

# =============================================
# Preprocessing Function
# =============================================
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# =============================================
# Custom CSS
# =============================================
st.markdown("""
    <style>
        .main { background-color: #0f0f0f; }
        .stTextArea textarea {
            background-color: #1a1a1a;
            color: #f0f0f0;
            border: 1px solid #333;
            border-radius: 8px;
            font-size: 16px;
        }
        .result-spam {
            background: linear-gradient(135deg, #ff4444, #cc0000);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-top: 20px;
        }
        .result-ham {
            background: linear-gradient(135deg, #00cc66, #009944);
            color: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================
# UI
# =============================================
st.title("📱 SMS / Email Spam Detector")
st.markdown("**Paste your message below and instantly know if it's spam or not.**")
st.markdown("---")

input_sms = st.text_area(
    "📝 Enter your message here:",
    placeholder="e.g. Congratulations! You've won a $1000 gift card. Click here to claim...",
    height=150
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("🔍 Analyze Message", use_container_width=True, type="primary")

if predict_btn:
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message first.")
    else:
        with st.spinner("Analyzing..."):
            # Preprocess
            transformed_sms = transform_text(input_sms)
            
            # Vectorize
            vector_input = vectorizer.transform([transformed_sms])
            
            # Predict
            result = model.predict(vector_input)[0]

        st.markdown("---")
        
        if result == 1:
            st.markdown('<div class="result-spam">🚨 SPAM DETECTED</div>', unsafe_allow_html=True)
            st.error("⚠️ This message appears to be **spam**. Do not click any links or share personal information.")
        else:
            st.markdown('<div class="result-ham">✅ NOT SPAM (Ham)</div>', unsafe_allow_html=True)
            st.success("✅ This message appears to be **legitimate**.")

        with st.expander("🔍 See Preprocessing Details"):
            st.write("**Original:**", input_sms)
            st.write("**After Preprocessing:**", transform_text(input_sms))

# =============================================
# Sidebar Info
# =============================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write("This app uses a **SVC** model trained on the SMS Spam Collection dataset.")
    st.markdown("---")
    st.write("**Pipeline:**")
    st.write("1. Text Cleaning")
    st.write("2. Tokenization")
    st.write("3. Stop Words Removal")
    st.write("4. Stemming (Porter)")
    st.write("5. TF-IDF Vectorization")
    st.write("6. SVC Classification")
    st.markdown("---")
    st.write("**Model:** SVC")
    st.write("**Dataset Source:** https://www.kaggle.com/uciml/sms-spam-collection-kaggle-version")
