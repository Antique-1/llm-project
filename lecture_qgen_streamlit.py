import streamlit as st
import os
import json
import random
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# ------------------------------------------
# 환경변수 로드
# ------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일에 키를 넣어주세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------
# 텍스트 전처리
# ------------------------------------------
def clean_text(text: str) -> str:
    return " ".join(text.split())

# ------------------------------------------
# Whisper 오디오 → 텍스트 변환
# ------------------------------------------
def transcribe_audio(audio_bytes, filename):
    try:
        with open(f"/tmp/{filename}", "wb") as f:
            f.write(audio_bytes)

        with open(f"/tmp/{filename}", "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        st.error(f"음성 변환 오류: {e}")
        return ""

# ------------------------------------------
# 강의 요약 생성
# ------------------------------------------
def summarize_text(text: str) -> str:
    prompt = f"""
다음 강의 내용을 한국어로 핵심 bullet 형식으로 간결하게 요약해줘.

### 강의 내용
{text}

### 요약:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"요약 오류: {e}")
        return ""


# ------------------------------------------
# 객관식 문제 생성
# ------------------------------------------
def generate_mcq(summary: str, n: int, difficulty: str) -> List[Dict[str, Any]]:
    prompt = f"""
다음 요약문을 기반으로 객관식 문제 {n}개를 만들어라.

조건:
- 각 문제는 question, choices(4개), answer(0~3), explanation 항목을 가진다.
- JSON 배열로만 출력한다.
- 보기는 서로 유사해야 한다.
- 난이도: {difficulty}

### 요약문
{summary}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.7,
        )
        text = response.choices[0].message.content

        # JSON만 추출
        start = text.find("[")
        end = text.rfind("]")
        json_str = text[start:end+1]

        questions = json.loads(json_str)

        # 보기 랜덤 셔플 + 정답 인덱스 재설정
        for q in questions:
            original_choices = q["choices"]
            correct = original_choices[q["answer"]]

            random.shuffle(q["choices"])
            q["answer"] = q["choices"].index(correct)

        return questions[:n]

    except Exception as e:
        st.error(f"문제 생성 오류: {e}")
        return []


# ------------------------------------------
# 해설 생성
# ------------------------------------------
def generate_explanation(question, choices, answer_idx):
    prompt = f"""
다음 객관식 문제에 대해 정답의 이유와 각 오답이 틀린 이유를 간단히 설명하라.

문제: {question}
보기:
{choices}
정답 인덱스: {answer_idx}
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=300,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except:
        return "해설 생성 실패"


# ------------------------------------------
# Streamlit 세션 초기화
# ------------------------------------------
if "summary" not in st.session_state:
    st.session_state.summary = None

if "mcqs" not in st.session_state:
    st.session_state.mcqs = None


# ------------------------------------------
# UI 구성
# ------------------------------------------
st.title("📘 인터넷 강의 자동 요약 & 문제 생성기")

st.markdown("텍스트 또는 오디오를 업로드하면 요약 + 객관식 문제를 자동 생성합니다.")

uploaded = st.file_uploader("텍스트(.txt) 또는 오디오 업로드(.mp3/.wav)", type=["txt", "mp3", "wav", "m4a"])

text_input = st.text_area("또는 텍스트 직접 입력", height=200)

difficulty = st.selectbox("난이도", ["초급", "중급", "고급"])
n_questions = st.slider("문제 개수", 1, 10, 5)
generate_btn = st.button("요약 및 문제 생성")


# ------------------------------------------
# 입력 처리
# ------------------------------------------
lecture_text = ""

if uploaded:
    if uploaded.name.endswith(".txt"):
        lecture_text = uploaded.read().decode("utf-8")
    else:
        st.info("오디오 파일 인식 중...")
        lecture_text = transcribe_audio(uploaded.read(), uploaded.name)

if text_input.strip():
    lecture_text = text_input.strip()


# ------------------------------------------
# 문제 생성 버튼
# ------------------------------------------
if generate_btn:
    if not lecture_text:
        st.warning("텍스트 또는 오디오를 입력해주세요.")
    else:
        st.session_state.summary = summarize_text(lecture_text)
        st.session_state.mcqs = generate_mcq(st.session_state.summary, n_questions, difficulty)

        # 해설 생성
        for q in st.session_state.mcqs:
            q["explanation"] = generate_explanation(q["question"], q["choices"], q["answer"])

        st.success("문제 생성 완료!")


# ------------------------------------------
# 결과 출력 (세션 유지)
# ------------------------------------------
if st.session_state.summary:
    with st.expander("📌 요약 보기 / 숨기기"):
        st.write(st.session_state.summary)

if st.session_state.mcqs:
    st.markdown("## 📝 생성된 문제")

    for i, q in enumerate(st.session_state.mcqs, 1):

        st.markdown(f"### 문제 {i}")
        st.write(q["question"])

        # 선택지 출력 + 세션 유지되는 라디오 버튼
        key = f"q_{i}_choice"
        st.session_state.setdefault(key, None)

        selected = st.radio(
            "정답 선택:",
            options=[f"{chr(65+j)}. {c}" for j, c in enumerate(q["choices"])],
            key=key
        )

        # 해설
        with st.expander("해설 보기"):
            correct = chr(65 + q["answer"])
            st.success(f"정답: {correct}")
            st.write(q["explanation"])

        st.markdown("---")
