"""Streamlit chat over the support KB. Conversational front-end (Week 2 bonus).
Run: python3 -m streamlit run src/app.py
(Remember to export your keys in the same terminal session.)
TODO (vibe session): add the product_area metadata filter as a dropdown; visual polish.
"""
import streamlit as st
from rag import answer_with_escalation

st.set_page_config(page_title="Support KB Assistant", page_icon=":speech_balloon:")
st.title("Support KB Assistant")
st.caption("Week 2 RAG — hybrid retrieval over a synthetic support KB + past tickets")

# Keep the conversation in session state so past turns stay on screen.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Re-draw the whole conversation each run.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for cid, text in msg["sources"]:
                    st.markdown(f"**[{cid}]** {text}")

# Chat input at the bottom.
if q := st.chat_input("Ask a support question"):
    # Show the user's message.
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    # Generate and show the assistant's reply.
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            result = answer_with_escalation(q)
            if result["status"] == "escalated":
                st.warning(result["message"])
                if result["closest_source"]:
                    st.caption(f"Closest article found: {result['closest_source']}")
                reply = result["message"]
                sources = []
            else:
                reply = result["message"]
                sources = result["sources"]
                st.markdown(reply)
                with st.expander("Sources used"):
                    for cid, text in sources:
                        st.markdown(f"**[{cid}]** {text}")
    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources}
    )