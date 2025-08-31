import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load a conversational model (lightweight)
chat_model = pipeline("text-generation", model="microsoft/DialoGPT-small")

# Function to caption image
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# Function to chat based on caption + user query
def chatbot_response(caption, user_input):
    context = f"The image shows: {caption}. User asked: {user_input}. Respond conversationally."
    reply = chat_model(context, max_length=100, num_return_sequences=1)
    return reply[0]["generated_text"].split("Respond conversationally.")[-1].strip()

# Streamlit UI
st.set_page_config(page_title="Vision Chatbot", page_icon="🤖")
st.title("🖼️ Vision + Chatbot")

if "caption" not in st.session_state:
    st.session_state.caption = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file and st.session_state.caption is None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Analyzing image..."):
        st.session_state.caption = generate_caption(image)
    st.success(f"AI sees: {st.session_state.caption}")

# Chat section (after image is uploaded)
if st.session_state.caption:
    user_input = st.text_input("💬 Ask something about the image:")
    if st.button("Send") and user_input:
        response = chatbot_response(st.session_state.caption, user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    # Display chat history
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**🧑 {speaker}:** {msg}")
        else:
            st.markdown(f"**🤖 {speaker}:** {msg}")
