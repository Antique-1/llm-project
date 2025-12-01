import streamlit as st
from typing import List, Dict, Any
import json
import os
import time
import re
from openai import OpenAI

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

SUMMARY_MODEL = "gpt-4o"
QA_MODEL = "gpt-4o"


# ----------------------------------------------------------------------------
# 유틸리티
# ----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------------------------------------------------------
# 오디오 -> 텍스트 변환 (예시)
# ----------------------------------------------------------------------------

def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """
    Whisper API 실제 사용 예시:
    with open("/tmp/audio.mp3","wb") as f:
        f.write(file_bytes)
    resp = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=open("/tmp/audio.mp3","rb")
    )
    return resp.text
    """
    return ""


# ----------------------------------------------------------------------------
# 요약 생성
# ----------------------------------------------------------------------------

def summarize_text(text: str, max_tokens: int = 512, model: str = SUMMARY_MODEL) -> str:
    prompt = (
        "다음 강의 스크립트를 한국어로 간결하게 요약하라. "
        "핵심 개념을 짧은 문장(불릿)으로 추출하라.\n\n"
        f"스크립트:\n{text}\n\n요약:"
    )

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=max_tokens,
            temperature=0.2
        )
        output = resp.output[0].content[0].text
        return output.strip()

    except Exception as e:
        st.error(f"요약 중 오류 발생: {e}")
        return ""


# ----------------------------------------------------------------------------
# 객관식 문제 생성
# ----------------------------------------------------------------------------

def generate_mcq_from_summary(summary: str, n_questions: int = 5, difficulty: str = "중급", model: str = QA_MODEL) -> List[Dict[str, Any]]:
    difficulty_map = {
        "초급": "핵심 용어 중심의 단순 문제",
        "중급": "개념 이해 + 간단 응용",
        "고급": "복합 개념 + 응용/추론"
    }
    diff_desc = difficulty_map.get(difficulty, difficulty_map["중급"])

    prompt = (
        "아래 요약문을 기반으로 한국어로 다지선다형 문제를 생성하라.\n"
        f"난이도: {difficulty}({diff_desc})\n\n"
        f"요약문:\n{summary}\n\n"
        "출력은 반드시 아래 형식의 JSON 배열로만 출력하라:\n"
        "[{\n"
        "  \"question\": \"...\",\n"
        "  \"choices\": [\"A\", \"B\", \"C\", \"D\"],\n"
        "  \"answer\": 0,\n"
        "  \"explanation\": \"...\"\n"
        "}]"
    )

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=1500,
            temperature=0.7
        )
        text = resp.output[0].content[0].text.strip()

        # JSON 부분 추출
        json_start = text.find("[")
        json_end = text.rfind("]")
        if json_start == -1 or json_end == -1:
            st.warning("JSON 추출 실패, 원문 출력:")
            st.code(text)
            return []

        json_text = text[json_start:json_end+1]

        try:
            mcqs = json.loads(json_text)
        except Exception as e:
            st.error(f"JSON 파싱 실패: {e}")
            st.code(json_text)
            return []

        valid = [item for item in mcqs if len(item.get("choices", [])) == 4]
        return valid[:n_questions]

    except Exception as e:
        st.error(f"문제 생성 오류: {e}")
        return []


# ----------------------------------------------------------------------------
# 문제 해설 생성
# ----------------------------------------------------------------------------

def generate_explanation(question: str, choices: List[str], answer_index: int, model: str = QA_MODEL) -> str:
    prompt = f"다음 문제에 대해 정답 근거와 오답 이유를 한국어로 간단히 설명하라.\n\n문제: {question}\n\n"
    for i, c in enumerate(choices):
        prompt += f"{i+1}. {c}\n"
    prompt += f"\n정답 번호: {answer_index+1}\n\n해설:"

    try:
        resp = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=300,
            temperature=0.2
        )
        return resp.output[0].content[0].text.strip()
    except Exception as e:
        return f"해설 생성 오류: {e}"


# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Lecture-QGen", layout="wide")
    st.title("인터넷 강의 자동 요약 및 객관식 문제 생성")

    st.sidebar.header("설정")
    n_questions = st.sidebar.number_input("문제 수", 1, 20, 5)
    difficulty = st.sidebar.selectbox("난이도", ["초급", "중급", "고급"], index=1)
    enable_explanations = st.sidebar.checkbox("해설 생성", True)

    st.markdown("### 1) 강의 텍스트 업로드 또는 직접 입력")
    uploaded = st.file_uploader("텍스트(.txt) / 오디오(.mp3/.wav) 업로드", type=["txt", "mp3", "wav", "m4a"])
    raw_text_area = st.text_area("또는 직접 텍스트 입력", height=200)

    lecture_text = ""
    if uploaded is not None:
        name = uploaded.name.lower()
        data = uploaded.read()

        if name.endswith(".txt"):
            lecture_text = data.decode("utf-8")
        else:
            st.info("오디오 파일 → 텍스트로 변환 중...")
            transcribed = transcribe_audio(data, name)
            if transcribed:
                lecture_text = transcribed
            else:
                st.warning("오디오 자동 변환이 설정되지 않음. 직접 텍스트 입력 필요.")

    if raw_text_area.strip():
        lecture_text = raw_text_area

    lecture_text = clean_text(lecture_text)

    if not lecture_text:
        st.info("강의 텍스트를 입력하세요.")
        return

    if st.button("요약 및 문제 생성 시작"):
        with st.spinner("요약 생성 중..."):
            summary = summarize_text(lecture_text)

        if not summary:
            st.error("요약 실패")
            return

        st.markdown("## 요약")
        st.write(summary)

        with st.spinner("문제 생성 중..."):
            mcqs = generate_mcq_from_summary(summary, n_questions, difficulty)

        if not mcqs:
            st.error("문제 생성 실패")
            return

        if enable_explanations:
            for item in mcqs:
                if not item.get("explanation"):
                    item["explanation"] = generate_explanation(
                        item["question"],
                        item["choices"],
                        int(item["answer"])
                    )

        st.markdown("## 생성된 문제")
        for i, q in enumerate(mcqs, 1):
            st.markdown(f"### {i}. {q['question']}")
            for idx, ch in enumerate(q["choices"]):
                mark = "(정답)" if idx == q["answer"] else ""
                st.write(f"{chr(65+idx)}. {ch} {mark}")

            if q.get("explanation"):
                with st.expander("해설"):
                    st.write(q["explanation"])

            st.markdown("---")

        result = {
            "summary": summary,
            "mcqs": mcqs,
            "meta": {
                "n_questions": n_questions,
                "difficulty": difficulty
            }
        }

        st.download_button(
            "JSON 다운로드",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="lecture_mcqs.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
