import os
import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db
import kb_updater
import multimodal_helper
from sentiment_helper import (
    analyze_sentiment,
    apply_sentiment_to_answer,
    sentiment_badge_html,
    get_sentiment_feedback,
)
from language_helper import (
    process_multilingual,
    translate_answer,
    get_greeting,
    SUPPORTED_LANGUAGES,
)

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING FUNCTIONALITY (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
st.title(" CUSTOMER SERVICE CHATBOT 🤖")

btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

# ─────────────────────────────────────────────────────────────────────────────
# TASK 6: Language Selector (above the question box)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("#### 🌍 Language Settings")

lang_options = {"auto": "🌐 Auto-Detect"} | {
    code: f"{info['flag']} {info['name']} ({info['native']})"
    for code, info in SUPPORTED_LANGUAGES.items()
}

selected_lang = st.selectbox(
    "Select language or let the chatbot detect it automatically:",
    options=list(lang_options.keys()),
    format_func=lambda x: lang_options[x],
    key="lang_selector"
)

# Show greeting in selected language
if selected_lang != "auto":
    st.caption(f"💬 {get_greeting(selected_lang)}")

# ─────────────────────────────────────────────────────────────────────────────
# Question Input
# ─────────────────────────────────────────────────────────────────────────────
question = st.text_input("Question: ")

if question:
    # ── TASK 6: Detect / Translate ────────────────────────────────────────────
    with st.spinner("Detecting language…"):
        lang_result = process_multilingual(question, manual_lang=selected_lang)

    detected    = lang_result["detected_lang"]
    lang_code   = detected["code"]
    eng_question = lang_result["english_question"]

    # Show detected language badge
    st.markdown(
        f"🌍 **Detected Language:** {detected['flag']} {detected['name']} ({detected['native']})"
    )
    if lang_code != "en":
        st.caption(f"🔄 Translated to English: *{eng_question}*")

    # ── TASK 5: Sentiment Analysis (on original question) ────────────────────
    sentiment = analyze_sentiment(question)
    st.markdown("**🧠 Detected Sentiment:**")
    st.markdown(sentiment_badge_html(sentiment), unsafe_allow_html=True)
    st.caption(get_sentiment_feedback(sentiment))

    # ── Core QA — use English version of the question ────────────────────────
    chain    = get_qa_chain()
    response = chain(eng_question)
    answer   = response["result"]

    # ── TASK 6: Translate answer back to user's language ─────────────────────
    if lang_code != "en":
        with st.spinner(f"Translating answer to {detected['name']}…"):
            answer = translate_answer(answer, lang_code)

    # ── TASK 5: Apply empathetic tone prefix ──────────────────────────────────
    answer_final = apply_sentiment_to_answer(answer, sentiment)

    st.header("Answer")
    st.write(answer_final)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1: Dynamic Knowledge Base Expansion
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📚 Knowledge Base Manager")

with st.expander("➕  Add New Data Manually", expanded=False):
    st.markdown(
        "Upload a CSV file with **`prompt`** and **`response`** columns "
        "to immediately extend the knowledge base."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv", key="kb_upload"
    )
    if uploaded_file:
        os.makedirs(kb_updater.PENDING_DIR, exist_ok=True)
        save_path = os.path.join(kb_updater.PENDING_DIR, uploaded_file.name)
        with open(save_path, "wb") as fh:
            fh.write(uploaded_file.getbuffer())
        st.info(f"File staged: `{uploaded_file.name}`")
        if st.button("⚡ Update Knowledge Base Now"):
            with st.spinner("Merging new data into the knowledge base …"):
                success, message = kb_updater.update_vector_db_from_file(save_path)
            if success:
                st.success(message)
            else:
                st.error(message)

with st.expander("⏰  Auto-Update Scheduler", expanded=False):
    st.markdown(
        "The scheduler watches the **`pending_updates/`** folder. "
        "Any CSV file you place there will be automatically merged "
        "into the knowledge base at the chosen interval."
    )
    interval = st.selectbox(
        "Check for new data every:",
        options=[1, 6, 12, 24],
        index=3,
        format_func=lambda x: f"{x} hour(s)",
        key="kb_interval"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶  Start Scheduler"):
            success, msg = kb_updater.start_scheduler(interval_hours=interval)
            st.success(msg) if success else st.warning(msg)
    with col2:
        if st.button("⏹  Stop Scheduler"):
            st.info(kb_updater.stop_scheduler())

    running = kb_updater.is_scheduler_running()
    st.markdown(
        f"**Scheduler status:** :{'green' if running else 'red'}[● {'Running' if running else 'Stopped'}]"
    )

    if st.button("🔄  Check Pending Updates Now"):
        with st.spinner("Scanning pending_updates/ folder …"):
            results = kb_updater.check_and_process_pending()
        if not results:
            st.info("No pending CSV files found.")
        else:
            for r in results:
                (st.success if r["success"] else st.error)(
                    f"{'✅' if r['success'] else '❌'}  {r['file']} — {r['message']}"
                )

with st.expander("📋  Update History", expanded=False):
    log = kb_updater.load_update_log()
    if not log:
        st.info("No updates logged yet.")
    else:
        for entry in reversed(log[-15:]):
            is_ok  = entry.get("status") == "success"
            badge  = ":green[✅ success]" if is_ok else ":red[❌ failed]"
            detail = (
                f"{entry.get('docs_added', 0)} docs added"
                if is_ok else entry.get("error", "")
            )
            st.markdown(
                f"**{entry.get('timestamp','—')}** &nbsp;|&nbsp; "
                f"`{os.path.basename(entry.get('source','—'))}` &nbsp;|&nbsp; "
                f"{badge} &nbsp;— {detail}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2: Multi-Modal Chatbot
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🖼️ Multi-Modal Chat")
st.markdown(
    "This section extends the chatbot with **image understanding** (Gemini Vision) "
    "and **image generation** (AI image synthesis)."
)

tab1, tab2 = st.tabs(["🔍 Understand an Image", "🎨 Generate an Image"])

with tab1:
    st.markdown(
        "Upload any image and ask a question about it. "
        "**Gemini 1.5 Flash** will analyse the image and respond."
    )
    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"], key="mm_image"
    )
    image_question = st.text_input(
        "Ask something about the image (leave blank for a general description):",
        key="mm_question"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("🔍 Analyse Image"):
            image_bytes = uploaded_image.read()
            with st.spinner("Gemini is analysing the image …"):
                result = multimodal_helper.multimodal_chat(
                    text_input=image_question,
                    image_bytes=image_bytes
                )
            st.markdown("### 📝 Gemini's Response")
            st.write(result["text_answer"])

with tab2:
    st.markdown(
        "Type what you want to see — the AI will generate an image for you."
    )
    gen_prompt = st.text_input(
        "Describe the image you want to generate:",
        placeholder="e.g. a data science workspace with Python code on screen",
        key="mm_gen_prompt"
    )
    col_w, col_h = st.columns(2)
    with col_w:
        img_width = st.select_slider(
            "Width (px)", options=[256, 512, 768], value=512, key="mm_width"
        )
    with col_h:
        img_height = st.select_slider(
            "Height (px)", options=[256, 512, 768], value=512, key="mm_height"
        )
    if st.button("🎨 Generate Image"):
        if not gen_prompt.strip():
            st.warning("Please enter a description first.")
        else:
            with st.spinner("Generating your image …"):
                img_bytes = multimodal_helper.generate_image(
                    prompt=gen_prompt, width=img_width, height=img_height
                )
            if img_bytes:
                st.image(img_bytes, caption=f"Generated: {gen_prompt}", use_column_width=True)
                st.download_button(
                    label="⬇️ Download Image",
                    data=img_bytes,
                    file_name="generated_image.png",
                    mime="image/png"
                )
            else:
                st.error("Image generation failed. Please try a different description.")

    if question and multimodal_helper.should_generate_image(question):
        st.info(
            f"💡 Detected image generation request. "
            f"Generating: **{multimodal_helper.extract_image_prompt(question)}**"
        )
        with st.spinner("Generating image from your question …"):
            img_bytes = multimodal_helper.generate_image(
                multimodal_helper.extract_image_prompt(question)
            )
        if img_bytes:
            st.image(img_bytes, use_column_width=True)
            st.download_button(
                label="⬇️ Download Image",
                data=img_bytes,
                file_name="generated_image.png",
                mime="image/png",
                key="mm_auto_download"
            )
