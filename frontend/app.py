import os, uuid
import requests
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="Weather Advisory Bot")
st.title("Weather Advisory Bot")
st.caption("Advice comes only from written SOPs. Ask e.g. 'Is it safe to bike to work in Bhopal today?'")

if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
    st.session_state.msgs = []

if st.sidebar.button("New session"):
    requests.post(f"{BACKEND}/reset/{st.session_state.sid}", timeout=10)
    st.session_state.sid, st.session_state.msgs = str(uuid.uuid4()), []

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("trace"):
            with st.expander("Why did it say that?"):
                st.json(m["trace"])

if prompt := st.chat_input("Ask about an outdoor activity..."):
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        r = requests.post(f"{BACKEND}/chat", json={"message": prompt, "session_id": st.session_state.sid}, timeout=90)
        r.raise_for_status()
        data = r.json()
        reply, trace = data["reply"], data["trace"]
    except Exception as e:
        reply, trace = f"Backend unreachable: {e}", None
    st.session_state.msgs.append({"role": "assistant", "content": reply, "trace": trace})
    with st.chat_message("assistant"):
        st.markdown(reply)
        if trace:
            with st.expander("Why did it say that?"):
                st.json(trace)
