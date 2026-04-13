import pickle
import re
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Spam Message Detector",
    layout="centered"
)

with open("spam.pkl", "rb") as f:
    classifier = pickle.load(f)


def normalize_tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


def get_word_token(item):
    if isinstance(item, (list, tuple, np.ndarray)) and len(item) > 0:
        return str(item[0]).lower()
    return str(item).lower()


def build_sample_vector(message, word_dict):
    tokens = normalize_tokens(message)
    sample = []

    for item in word_dict:
        token = get_word_token(item)
        sample.append(tokens.count(token))

    return np.array(sample).reshape(1, 3000)


def get_spam_percent(sample):
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(sample)[0]
        return round(float(probabilities[1]) * 100, 2)

    if hasattr(classifier, "decision_function"):
        score = float(classifier.decision_function(sample)[0])
        probability = 1 / (1 + np.exp(-score))
        return round(probability * 100, 2)

    prediction = int(classifier.predict(sample)[0])
    return 100.0 if prediction == 1 else 0.0


def get_detection_details(message):
    details = []
    lower_msg = message.lower()

    if re.search(r"https?://|www\.", lower_msg):
        details.append("contains URLs")

    if re.search(r"\b\d+%\b", lower_msg):
        details.append("contains percentage offer")

    if any(word in lower_msg for word in ["discount", "offer", "promo", "coupon", "sale"]):
        details.append("contains promotional wording")

    if any(word in lower_msg for word in ["code", "voucher", "coupon"]):
        details.append("contains code-related wording")

    if any(word in lower_msg for word in ["urgent", "limited", "expires", "deadline", "midnight"]):
        details.append("contains urgency wording")

    if any(word in lower_msg for word in ["click", "claim", "join", "subscribe", "buy now"]):
        details.append("contains action-trigger wording")

    if lower_msg.count("!") >= 2:
        details.append("contains repeated exclamation marks")

    uppercase_words = re.findall(r"\b[A-Z]{3,}\b", message)
    if len(uppercase_words) >= 2:
        details.append("contains excessive uppercase words")

    if re.search(r"\b\d{4,}\b", message):
        details.append("contains numeric code or reference")

    return details


def render_result_box(label, color, spam_percent, details):
    spam_percent = round(float(spam_percent), 2)

    if color == "green":
        border = "#A7F3D0"
        text_color = "#16A34A"
        fill_color = "#22C55E"
        bg = "#F0FDF4"
        icon = "✅"
    elif color == "yellow":
        border = "#FDE68A"
        text_color = "#CA8A04"
        fill_color = "#EAB308"
        bg = "#FFFBEB"
        icon = "⚠️"
    else:
        border = "#FCA5A5"
        text_color = "#DC2626"
        fill_color = "#EF4444"
        bg = "#FEF2F2"
        icon = "⛔"

    if details:
        details_html = "".join(
            f"<div style='font-size:15px; color:#4B5563; margin-bottom:6px;'>• {item}</div>"
            for item in details
        )
    else:
        details_html = (
            "<div style='font-size:15px; color:#4B5563; margin-bottom:6px;'>"
            "• no obvious spam indicators found"
            "</div>"
        )

    box_html = f"""
<div style="border:2px solid {border}; background:{bg}; border-radius:16px; padding:24px; font-family:Arial, sans-serif;">
  <div style="font-size:24px; font-weight:700; color:{text_color}; margin-bottom:20px;">{icon} {label}</div>
  <div style="font-size:16px; font-weight:600; color:#4B5563; margin-bottom:8px;">Spam Confidence</div>
  <div style="width:100%; background:#E5E7EB; height:14px; border-radius:999px; overflow:hidden; margin-bottom:8px;">
    <div style="width:{spam_percent}%; background:{fill_color}; height:14px; border-radius:999px;"></div>
  </div>
  <div style="text-align:right; font-size:16px; font-weight:700; color:{text_color}; margin-bottom:20px;">{spam_percent}%</div>
  <div style="font-size:16px; font-weight:600; color:#4B5563; margin-bottom:10px;">Detection Details:</div>
  {details_html}
</div>
"""
    height = 230 + (len(details) * 28 if details else 28)
    components.html(box_html, height=height, scrolling=False)


def main():
    st.markdown(
        """
<style>
.stTextArea textarea {
    border-radius: 14px !important;
    border: 2px solid #93C5FD !important;
    font-size: 16px !important;
}
.stButton > button {
    width: 100%;
    border-radius: 12px !important;
    background-color: #2563EB !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 12px 16px !important;
}
.stButton > button:hover {
    background-color: #1D4ED8 !important;
    color: white !important;
}
</style>
        """,
        unsafe_allow_html=True
    )

    st.title("Spam Message Detector")
    st.write("Built with Streamlit & Python")

    message = st.text_area("Enter a text :", height=220)

    if st.button("Click to Analyze Message"):
        if message is None or message.strip() == "":
            st.error("Invalid input")
            st.stop()

        with open("word_pickle.pkl", "rb") as f:
            word_dict = pickle.load(f)

        sample = build_sample_vector(message, word_dict)
        spam_percent = get_spam_percent(sample)
        details = get_detection_details(message)

        if 45 < spam_percent < 75:
            render_result_box("Suspicious", "yellow", spam_percent, details)
        elif spam_percent >= 75:
            render_result_box("Spam Message", "red", spam_percent, details)
        else:
            render_result_box("Safe Message", "green", spam_percent, details)


main()