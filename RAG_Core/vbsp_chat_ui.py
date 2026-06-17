# RAG_Core/vbsp_chat_ui.py
"""
Giao diện Streamlit cho VBSP Internal RAG Chatbot
Chạy: streamlit run vbsp_chat_ui.py
Yêu cầu: pip install streamlit requests
"""

import streamlit as st
import requests
import json
import time
import re
from typing import Generator

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8522"
PAGE_TITLE   = "Trợ lý ảo VBSP"
BANK_NAME    = "Ngân hàng Chính sách Xã hội Việt Nam"

QUICK_PROMPTS = [
    "Điều kiện vay vốn hộ nghèo?",
    "Quy trình giải ngân cho vay?",
    "Lãi suất cho vay ưu đãi hiện hành?",
    "Hồ sơ vay vốn gồm những gì?",
]

# ─────────────────────────────────────────────────────────────
# CSS — override Streamlit native chat components only
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

/* Base */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"] > div {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    background: #F2F5F9 !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none !important; }
.block-container {
    padding: 0 0 80px 0 !important;
    max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #00287A 0%, #003D99 55%, #0050CC 100%) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #D6E4FF !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    color: #fff !important; border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.1) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.22) !important;
    border-radius: 8px !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 0.83rem !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.2) !important;
}

/* ── Chat message bubbles (native st.chat_message) ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 4px 0 !important;
    gap: 10px !important;
}

/* Avatar */
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
    width: 36px !important; height: 36px !important;
    border-radius: 50% !important;
    font-size: 1rem !important;
    flex-shrink: 0 !important;
}
[data-testid="stChatMessageAvatarAssistant"] {
    background: #003D99 !important;
}
[data-testid="stChatMessageAvatarUser"] {
    background: #E8F0FF !important;
}

/* Content area of each message */
[data-testid="stChatMessageContent"] {
    background: transparent !important;
    padding: 0 !important;
}

/* Bot bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"])
[data-testid="stChatMessageContent"] > div:first-child {
    background: #FFFFFF !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 13px 17px !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07) !important;
    color: #1A2A4A !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    max-width: 75% !important;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] > div:first-child {
    background: linear-gradient(135deg, #0055CC, #003D99) !important;
    border-radius: 18px 4px 18px 18px !important;
    padding: 11px 17px !important;
    color: #fff !important;
    font-size: 0.92rem !important;
    line-height: 1.65 !important;
    max-width: 68% !important;
    margin-left: auto !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] > div:first-child p { color: #fff !important; }

/* Markdown inside bubbles */
[data-testid="stChatMessageContent"] p { margin: 0 0 6px !important; }
[data-testid="stChatMessageContent"] p:last-child { margin-bottom: 0 !important; }
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol {
    margin: 4px 0 6px 18px !important; padding: 0 !important;
}
[data-testid="stChatMessageContent"] li { margin-bottom: 3px !important; }
[data-testid="stChatMessageContent"] strong { font-weight: 600 !important; }
[data-testid="stChatMessageContent"] a {
    color: #0055CC !important; text-decoration: underline !important;
}

/* ── References card ── */
.ref-card {
    margin-top: 10px;
    background: #F0F4FB;
    border-radius: 10px;
    padding: 10px 14px;
    border-left: 3px solid #0055CC;
    font-family: 'Be Vietnam Pro', sans-serif;
}
.ref-title {
    font-size: 0.7rem; font-weight: 700;
    color: #0055CC; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 7px;
}
.ref-item {
    font-size: 0.8rem; color: #3A5A8A;
    padding: 4px 0; display: flex; gap: 8px;
    align-items: flex-start; line-height: 1.45;
    border-bottom: 1px solid rgba(0,61,153,0.06);
}
.ref-item:last-child { border-bottom: none; }
.ref-item a { color: #0055CC; text-decoration: none; }
.ref-item a:hover { text-decoration: underline; }
.ref-badge {
    background: #E8F0FF; color: #003D99;
    font-size: 0.66rem; font-weight: 700;
    padding: 2px 7px; border-radius: 4px;
    flex-shrink: 0; margin-top: 1px;
    letter-spacing: 0.03em;
}
.ref-page {
    color: #8FA8C8; font-size: 0.74rem;
    white-space: nowrap; flex-shrink: 0;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    font-size: 0.9rem !important;
    border-radius: 14px !important;
    border: 1.5px solid #C5D5EF !important;
    background: #fff !important;
}
[data-testid="stChatInput"]:focus-within textarea {
    border-color: #0055CC !important;
    box-shadow: 0 0 0 3px rgba(0,85,204,0.1) !important;
}

/* ── Quick-prompt buttons ── */
div[data-testid="column"] .stButton > button {
    background: #fff !important;
    color: #0055CC !important;
    border: 1.5px solid #C5D5EF !important;
    border-radius: 20px !important;
    font-size: 0.8rem !important;
    padding: 6px 14px !important;
    font-family: 'Be Vietnam Pro', sans-serif !important;
    transition: all 0.15s !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: #0055CC !important;
    color: #fff !important;
    border-color: #0055CC !important;
}

/* ── Timestamp ── */
.msg-ts {
    font-size: 0.67rem; color: #9BA8BB;
    margin-top: 3px; padding: 0 2px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #C5D5EF; border-radius: 2px; }
</style>
"""

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=4)
        return r.status_code == 200 and r.json().get("status") in ("healthy", "degraded")
    except Exception:
        return False


def stream_chat(question: str, history: list) -> Generator:
    """Yield (type, payload) tuples from the SSE stream."""
    payload = {"question": question, "history": history, "stream": True}
    try:
        with requests.post(
            f"{API_BASE_URL}/chat", json=payload, stream=True, timeout=90
        ) as resp:
            if resp.status_code != 200:
                yield ("error", f"API lỗi {resp.status_code}")
                return
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                t = data.get("type")
                if t == "chunk":
                    yield ("chunk", data.get("content", ""))
                elif t == "references":
                    yield ("references", data.get("references", []))
                elif t == "error":
                    yield ("error", data.get("content", "Lỗi không xác định"))
    except requests.exceptions.ConnectionError:
        yield ("error", "Không thể kết nối tới API. Kiểm tra server đang chạy chưa.")
    except requests.exceptions.Timeout:
        yield ("error", "API timeout. Vui lòng thử lại.")
    except Exception as e:
        yield ("error", str(e))


def clean_filename(raw: str) -> str:
    """Extract readable filename from document_id or description."""
    if not raw:
        return "Tài liệu"
    # Remove path prefix
    name = raw.split("/")[-1]
    # Remove extension
    name = re.sub(r'\.(docx|pdf|xlsx?|pptx?|txt)$', '', name, flags=re.I)
    # Replace underscores/hyphens
    name = name.replace("_", " ").replace("-", " ")
    # Truncate
    return name[:60] if len(name) > 60 else name


def build_ref_html(refs: list) -> str:
    """Build a clean references card HTML.
    Lưu ý: KHÔNG thụt lề / xuống dòng trong chuỗi HTML — Markdown coi dòng
    thụt lề >=4 spaces là code block, sẽ làm lộ HTML thô ra UI."""
    if not refs:
        return ""

    items_html = ""
    seen = set()
    for ref in refs:
        doc_id  = ref.get("document_id", "")
        badge   = ref.get("type", "DOC")
        url     = ref.get("url", "")
        fname   = ref.get("filename", "") or ref.get("description", "") or doc_id
        page    = ref.get("page_num")
        section = ref.get("section_path", "")

        key = doc_id or fname
        if key in seen:
            continue
        seen.add(key)

        display = clean_filename(fname) or clean_filename(doc_id) or "Tài liệu"

        meta_parts = []
        if section:
            meta_parts.append(section[:40])
        if page:
            meta_parts.append(f"trang {page}")
        page_tag = f'<span class="ref-page">· {" · ".join(meta_parts)}</span>' if meta_parts else ""

        if url:
            label = f'<a href="{url}" target="_blank" title="{display}">{display}</a>'
        else:
            label = f'<span title="{doc_id}">{display}</span>'

        # Một dòng duy nhất, không có khoảng trắng đầu dòng
        items_html += (
            f'<div class="ref-item">'
            f'<span class="ref-badge">{badge}</span>'
            f'{label}{page_tag}'
            f'</div>'
        )

    if not items_html:
        return ""

    return (
        f'<div class="ref-card">'
        f'<div class="ref-title">📎 Nguồn tham khảo</div>'
        f'{items_html}'
        f'</div>'
    )

def fmt_time() -> str:
    return time.strftime("%H:%M")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title=PAGE_TITLE, page_icon="🏦",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []   # {role, content, refs, ts}
    if "api_ok" not in st.session_state:
        st.session_state.api_ok = False
    if "pending" not in st.session_state:
        st.session_state.pending = None

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.08);border-radius:12px;
                    padding:18px 16px 14px;text-align:center;margin-bottom:4px">
            <div style="font-size:2rem">🏦</div>
            <div style="font-weight:700;font-size:1rem;color:#fff;margin-top:6px">
                Trợ lý ảo VBSP</div>
            <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);margin-top:3px">
                Hệ thống nội bộ</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Status
        st.markdown(
            '<p style="font-size:0.68rem;font-weight:700;letter-spacing:.1em;'
            'text-transform:uppercase;color:rgba(255,255,255,0.45);margin:0 0 6px">Trạng thái</p>',
            unsafe_allow_html=True
        )
        col_s, col_b = st.columns([3, 2])
        with col_s:
            dot = "🟢" if st.session_state.api_ok else "🔴"
            label = "Hoạt động" if st.session_state.api_ok else "Chưa kết nối"
            st.markdown(
                f'<div style="font-size:0.82rem;padding-top:6px">{dot} {label}</div>',
                unsafe_allow_html=True
            )
        with col_b:
            if st.button("Kiểm tra", key="btn_health"):
                st.session_state.api_ok = check_api_health()
                st.rerun()

        st.markdown("---")

        # Stats
        n = len(st.session_state.messages)
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.08);border-radius:8px;'
            f'padding:10px 14px;font-size:0.82rem;color:rgba(255,255,255,0.8)">'
            f'💬 <strong>{n}</strong> tin nhắn trong phiên</div>',
            unsafe_allow_html=True
        )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🗑️  Xóa lịch sử hội thoại", key="btn_clear"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.markdown(
            f'<div style="font-size:0.78rem;color:rgba(255,255,255,0.55);line-height:1.6">'
            f'<strong style="color:rgba(255,255,255,0.8)">{BANK_NAME}</strong><br>'
            f'Hỗ trợ nghiệp vụ, quy trình,<br>chính sách tín dụng nội bộ.</div>',
            unsafe_allow_html=True
        )

    # ── Top bar ──────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#00287A,#0055CC);
                padding:16px 28px;display:flex;align-items:center;gap:14px;
                box-shadow:0 2px 12px rgba(0,40,122,.2)">
        <div style="background:#fff;border-radius:10px;width:40px;height:40px;
                    display:flex;align-items:center;justify-content:center;
                    font-size:1.3rem;flex-shrink:0">🏦</div>
        <div>
            <div style="color:#fff;font-weight:700;font-size:1.1rem;
                        font-family:'Be Vietnam Pro',sans-serif">Trợ lý ảo VBSP</div>
            <div style="color:rgba(255,255,255,.65);font-size:.76rem;margin-top:2px;
                        font-family:'Be Vietnam Pro',sans-serif">
                Ngân hàng Chính sách Xã hội Việt Nam · Hệ thống nội bộ</div>
        </div>
        <div style="margin-left:auto;width:9px;height:9px;border-radius:50%;
                    background:#2ECC71;box-shadow:0 0 0 3px rgba(46,204,113,.25)"></div>
    </div>
    """, unsafe_allow_html=True)

    # Auto health check on first load
    if not st.session_state.api_ok:
        st.session_state.api_ok = check_api_health()

    # ── Render history ────────────────────────────────────────
    if not st.session_state.messages:
        # Welcome
        st.markdown("""
        <div style="text-align:center;padding:52px 24px 28px;color:#4A6080">
            <div style="font-size:3rem;margin-bottom:14px">🏦</div>
            <h2 style="color:#003D99;font-size:1.2rem;font-weight:700;
                       font-family:'Be Vietnam Pro',sans-serif;margin-bottom:8px">
                Xin chào! Tôi có thể giúp gì cho bạn?</h2>
            <p style="font-size:0.88rem;line-height:1.65;max-width:400px;
                      margin:0 auto 6px;font-family:'Be Vietnam Pro',sans-serif">
                Trợ lý ảo nội bộ của <strong>Ngân hàng Chính sách Xã hội Việt Nam</strong>,
                sẵn sàng hỗ trợ về nghiệp vụ, quy trình và quy định nội bộ VBSP.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<p style="text-align:center;font-size:.8rem;color:#7A90AA;margin-bottom:10px">'
            'Gợi ý câu hỏi:</p>',
            unsafe_allow_html=True
        )
        cols = st.columns(2)
        for i, prompt in enumerate(QUICK_PROMPTS):
            with cols[i % 2]:
                if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                    st.session_state.pending = prompt
                    st.rerun()
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            avatar = "🏦" if role == "assistant" else "👤"
            with st.chat_message(role, avatar=avatar):
                # Render markdown content (bold, lists, links all work natively)
                st.markdown(msg["content"])

                # References card (HTML, separate from markdown bubble)
                refs_html = build_ref_html(msg.get("refs", []))
                if refs_html:
                    st.markdown(refs_html, unsafe_allow_html=True)

                # Timestamp
                if msg.get("ts"):
                    st.markdown(
                        f'<div class="msg-ts">{msg["ts"]}</div>',
                        unsafe_allow_html=True
                    )

    # ── Input bar ────────────────────────────────────────────
    placeholder = (
        "Nhập câu hỏi về nghiệp vụ VBSP..."
        if st.session_state.api_ok
        else "⚠️  API chưa kết nối — nhấn 'Kiểm tra' ở thanh bên"
    )
    user_input = st.chat_input(placeholder, disabled=not st.session_state.api_ok)

    # Resolve question source
    question = None
    if st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None
    elif user_input:
        question = user_input.strip()

    # ── Process & stream ──────────────────────────────────────
    if question:
        ts_user = fmt_time()

        # Append user message
        st.session_state.messages.append({
            "role": "user", "content": question, "ts": ts_user
        })

        # Show user bubble immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
            st.markdown(f'<div class="msg-ts">{ts_user}</div>', unsafe_allow_html=True)

        # Stream bot response
        with st.chat_message("assistant", avatar="🏦"):
            text_placeholder = st.empty()
            full_text  = ""
            references = []
            error_msg  = None

            # Build API history (last 20 messages)
            api_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-20:]
                if m["role"] in ("user", "assistant")
            ]

            for ev_type, payload in stream_chat(question, api_history):
                if ev_type == "chunk":
                    full_text += payload
                    text_placeholder.markdown(full_text + "▌")
                elif ev_type == "references":
                    references = payload
                elif ev_type == "error":
                    error_msg = payload

            # Remove cursor, final render
            if error_msg and not full_text:
                full_text = f"⚠️ {error_msg}"

            text_placeholder.markdown(full_text)

            # References below bubble
            refs_html = build_ref_html(references)
            if refs_html:
                st.markdown(refs_html, unsafe_allow_html=True)

            ts_bot = fmt_time()
            st.markdown(f'<div class="msg-ts">{ts_bot}</div>', unsafe_allow_html=True)

        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "refs": references,
            "ts": ts_bot,
        })
        st.rerun()


if __name__ == "__main__":
    main()