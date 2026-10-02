import json
import logging
import os

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-6-astra"
REQUEST_TIMEOUT = 30  # 초
MAX_RETRIES = 1

MAX_INPUT_CHARS = 5000

# 대상 언어: JSON 키 -> (화면 표시 이름, 프롬프트용 언어명)
# 국기 이모지는 Windows에서 "US"처럼 글자로 보이므로 언어 이름으로 표기한다.
TARGET_LANGUAGES = {
    "english": ("English · 영어", "English"),
    "japanese": ("日本語 · 일본어", "Japanese"),
    "vietnamese": ("Tiếng Việt · 베트남어", "Vietnamese"),
}

SYSTEM_PROMPT = (
    "You are a professional translator. Detect the language of the user's text "
    "and translate it naturally into the following languages, preserving meaning, "
    "tone, and line breaks: {languages}.\n"
    "Respond ONLY with a JSON object in this exact format, with no extra text:\n"
    "{schema}"
)

# 예외 타입 -> 사용자 안내 문구 (위에서부터 먼저 일치하는 항목 사용)
ERROR_MESSAGES = [
    (openai.AuthenticationError, "API 키가 올바르지 않습니다. `.env` 또는 Streamlit Secrets의 OPENAI_API_KEY를 확인해 주세요."),
    (openai.PermissionDeniedError, "이 API 키로는 요청한 기능을 사용할 수 없습니다. 계정 권한을 확인해 주세요."),
    (openai.NotFoundError, "설정된 모델({model})을 찾을 수 없습니다. OPENAI_MODEL 값을 확인해 주세요."),
    (openai.RateLimitError, "요청 한도 또는 사용량을 초과했습니다. 잠시 후 다시 시도하거나 결제 상태를 확인해 주세요."),
    (openai.APITimeoutError, "응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요."),
    (openai.APIConnectionError, "번역 서버에 연결하지 못했습니다. 인터넷 연결을 확인해 주세요."),
    (openai.BadRequestError, "요청을 처리하지 못했습니다. 입력이 너무 길거나 모델이 이 요청 형식을 지원하지 않을 수 있습니다."),
    (openai.InternalServerError, "번역 서버에 일시적인 문제가 있습니다. 잠시 후 다시 시도해 주세요."),
    (json.JSONDecodeError, "번역 결과를 읽지 못했습니다. 다시 시도해 주세요."),
    (ValueError, "번역 결과가 일부 누락되었습니다. 다시 시도해 주세요."),
]


class TranslationError(Exception):
    """사용자에게 보여줄 안내 문구와 원본 예외를 함께 담는다."""

    def __init__(self, message: str, detail: Exception):
        super().__init__(message)
        self.detail = detail


def get_setting(key: str, default: str | None = None) -> str | None:
    """.env(환경 변수)를 먼저 읽고, 없으면 Streamlit Secrets에서 읽는다."""
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


# 같은 입력은 캐시된 결과를 사용한다.
# 버튼을 연타해도 Streamlit이 재실행을 순차 처리하므로 첫 호출 이후에는 캐시가 응답한다.
@st.cache_data(show_spinner=False, max_entries=256, ttl=3600)
def translate(api_key: str, model: str, text: str) -> dict[str, str]:
    client = openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
    languages = ", ".join(name for _, name in TARGET_LANGUAGES.values())
    schema = json.dumps({key: "..." for key in TARGET_LANGUAGES})

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(languages=languages, schema=schema)},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    missing = [key for key in TARGET_LANGUAGES if not isinstance(result.get(key), str)]
    if missing:
        raise ValueError(f"응답에 누락된 언어가 있습니다: {', '.join(missing)}")
    return result


def run_translation(api_key: str, model: str, text: str) -> dict[str, str]:
    try:
        return translate(api_key, model, text)
    except Exception as e:
        for error_type, message in ERROR_MESSAGES:
            if isinstance(e, error_type):
                logger.warning("Translation failed: %r", e)
                raise TranslationError(message.format(model=model), e) from e
        logger.exception("Unexpected translation error")
        raise TranslationError("알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", e) from e


def clear_input() -> None:
    st.session_state.text = ""
    st.session_state.pop("result", None)
    st.session_state.pop("source", None)


def main() -> None:
    st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="wide")
    # 좁은 화면에서 제목이 "번역 / 기"처럼 글자 단위로 끊기지 않도록 단어 단위 줄바꿈
    st.html("<style>h1 { word-break: keep-all; }</style>")
    st.title("🌐 다국어 번역기")
    st.caption("입력한 글을 영어 · 일본어 · 베트남어로 번역합니다.")

    api_key = get_setting("OPENAI_API_KEY")
    model = get_setting("OPENAI_MODEL", DEFAULT_MODEL)

    if not api_key:
        st.error(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "로컬에서는 `.env` 파일에, 배포 환경에서는 Streamlit Secrets에 키를 등록하세요."
        )
        st.stop()

    text = st.text_area(
        "번역할 텍스트",
        key="text",
        height=200,
        placeholder=f"번역할 텍스트를 입력하세요... (최대 {MAX_INPUT_CHARS:,}자)",
    )
    # st.text_area(max_chars=...)는 한도를 넘는 붙여넣기를 안내 없이 무시하므로 직접 검사한다
    over_limit = len(text) > MAX_INPUT_CHARS
    counter = f"{len(text):,} / {MAX_INPUT_CHARS:,}자"

    # 글자 수는 왼쪽, 버튼은 오른쪽에 한 줄로 배치
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(f":red[{counter} · {len(text) - MAX_INPUT_CHARS:,}자 초과]" if over_limit else counter)
        st.space("stretch")
        st.button("지우기", icon=":material/close:", on_click=clear_input, disabled=not text)
        translate_clicked = st.button("번역하기", type="primary", icon=":material/translate:")

    if translate_clicked:
        # 새 요청을 시작하면 이전 결과는 버린다 (실패 시 옛 결과가 다시 보이지 않도록)
        st.session_state.pop("result", None)
        st.session_state.pop("source", None)

        if not text.strip():
            st.warning("번역할 텍스트를 입력해 주세요.")
        elif over_limit:
            st.warning(f"한 번에 최대 {MAX_INPUT_CHARS:,}자까지 번역할 수 있습니다. 텍스트를 줄여 주세요.")
        else:
            try:
                with st.spinner("번역 중입니다..."):
                    st.session_state.result = run_translation(api_key, model, text)
                    st.session_state.source = text
            except TranslationError as e:
                st.error(str(e))
                with st.expander("자세한 오류 내용"):
                    st.code(f"{type(e.detail).__name__}: {e.detail}", language=None, wrap_lines=True)

    st.divider()
    result = st.session_state.get("result")
    if result and text != st.session_state.get("source"):
        st.info("입력이 변경되었습니다. 아래는 이전 입력의 번역 결과입니다. [번역하기]를 눌러 다시 번역해 주세요.")

    # 좁은 화면에서는 Streamlit이 컬럼을 자동으로 세로로 쌓는다
    for column, (key, (label, _)) in zip(st.columns(len(TARGET_LANGUAGES)), TARGET_LANGUAGES.items()):
        with column, st.container(border=True):
            st.markdown(f"**{label}**")
            if result:
                # st.code는 우측 상단에 복사 버튼을 제공한다
                st.code(result[key], language=None, wrap_lines=True)
            else:
                st.caption("번역 결과가 여기에 표시됩니다.")


if __name__ == "__main__":
    main()
