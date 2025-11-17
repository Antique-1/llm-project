import streamlit as st
from typing import List, Dict, Any
import json
import os
import time
import openai
import re

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# 기본 모델 설정
SUMMARY_MODEL = "gpt-4o"  
QA_MODEL = "gpt-4o"

# ----------------------------------------------------------------------------
# 유틸리티 함수
# ----------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """간단한 전처리: 중복 공백 제거, 특수문자 정리 등."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

# ----------------------------------------------------------------------------
# 오디오 -> 텍스트 변환
# 사용자가 텍스트만 업로드한다면 이 부분은 통과됩니다.
# 실제 구현에서는 Whisper, OpenAI 음성 API, 또는 클라이언트 측 전처리를 권장.
# ----------------------------------------------------------------------------

def transcribe_audio(file_bytes: bytes, filename: str) -> str:
    """간단한 placeholder 함수: 사용자가 Whisper나 OpenAI 음성 API를 연결하도록 안내
    실제로는 파일을 임시로 저장하고 whisper로 전송하여 텍스트를 반환.
    """
    # TODO: Whisper/WhisperX/OpenAI audio transcription 연동
    return ""  # 빈 문자열 반환하면 호출부에서 텍스트 업로드 필요

# ----------------------------------------------------------------------------
# 요약 함수
# ----------------------------------------------------------------------------

def summarize_text(text: str, max_tokens: int = 512, model: str = SUMMARY_MODEL) -> str:
    """텍스트를 요약하여 핵심 문장/개념을 반환합니다.
    간단한 프롬프트 템플릿 사용
    """
    prompt = f"다음 강의 스크립트를 한국어로 간결하게 요약하라. 핵심 개념을 짧은 문장(불릿)으로 추출하라.\n\n스크립트:\n{text}\n\n요약:" 
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        summary = resp["choices"][0]["message"]["content"].strip()
        return summary
    except Exception as e:
        st.error(f"요약 중 오류가 발생했습니다: {e}")
        return ""

# ----------------------------------------------------------------------------
# 객관식 문제(MCQ) 생성 함수
# ----------------------------------------------------------------------------

def generate_mcq_from_summary(summary: str, n_questions: int = 5, difficulty: str = "중급", model: str = QA_MODEL) -> List[Dict[str, Any]]:
    """요약문(또는 원문 기반)으로부터 다지선다형 문제를 생성.
    반환 형식: [{question:, choices:[...], answer: index, explanation: str}, ...]
    """
    # 난이도에 따른 프롬프트 세부 조정(예시)
    difficulty_map = {
        "초급": "객관식 문제는 핵심 용어 확인 수준(직접적이고 단순한 보기)",
        "중급": "개념 이해 및 간단한 응용이 가능한 수준",
        "고급": "판단력, 두 개 이상의 개념 결합, 부분 계산/논리 필요 수준",
    }
    diff_desc = difficulty_map.get(difficulty, difficulty_map["중급"])

    prompt = (
        "아래 요약문을 기반으로 한국어로 다지선다형 객관식 문제를 생성하라."
        f"\n난이도: {difficulty}({diff_desc})"
        f"\n요약문:\n{summary}\n\n"
        "요청 형식: JSON 배열. 각 항목은 question(문장), choices(리스트, 4개), answer(정답 인덱스 0-3), explanation(해설)")
    

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            max_tokens=1200,
            temperature=0.7,
        )
        content = resp["choices"][0]["message"]["content"].strip()

        # 모델이 JSON을 반환했다고 가정하고 파싱 시도
        try:
            # 종종 모델 출력 앞뒤에 설명 텍스트가 붙음 -> JSON 부분만 추출
            json_start = content.find("[")
            json_end = content.rfind("]")
            if json_start != -1 and json_end != -1:
                json_text = content[json_start:json_end+1]
                mcqs = json.loads(json_text)
            else:
                # 파싱 실패 시 빈 리스트 반환
                st.warning("MCQ 생성 결과에서 JSON을 찾을 수 없습니다. 원시 출력을 아래에 표시합니다.")
                st.code(content)
                mcqs = []
        except Exception as e:
            st.error(f"MCQ 파싱 실패: {e}")
            st.code(content)
            mcqs = []

        # 기본 검증: 각 문제에 choices가 4개인지 확인
        valid_mcqs = []
        for item in mcqs:
            try:
                if len(item.get("choices", [])) == 4:
                    valid_mcqs.append(item)
                else:
                    # 보기가 4개가 아닌 경우 보완 시도(간단한 방법)
                    st.warning(f"문제의 보기가 4개가 아니어서 제외: {item.get('question')}")
            except Exception:
                continue

        return valid_mcqs[:n_questions]

    except Exception as e:
        st.error(f"MCQ 생성 중 오류: {e}")
        return []

# ----------------------------------------------------------------------------
# 해설(오답 설명) 자동 생성(문제별)
# ----------------------------------------------------------------------------

def generate_explanation(question: str, choices: List[str], answer_index: int, model: str = QA_MODEL) -> str:
    """정답 이유와 각 오답에 대한 간단한 오류 설명을 생성"""
    prompt = (
        f"다음 문제에 대해 정답의 근거와 각 보기(오답)의 오류를 한국어로 간단히 설명하라.\n\n"
        f"문제: {question}\n"
        f"보기:\n"
    )
    for i, c in enumerate(choices):
        prompt += f"{i+1}. {c}\n"
    prompt += f"정답 번호: {answer_index+1}\n\n출력 형식: 간단한 해설 텍스트"

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"해설 생성 오류: {e}"

# ----------------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Lecture-QGen", layout="wide")
    st.title("인터넷 강의 자동 요약 및 객관식 문제 생성")

    st.sidebar.header("설정")
    n_questions = st.sidebar.number_input("생성할 문제 수", min_value=1, max_value=20, value=5)
    difficulty = st.sidebar.selectbox("난이도", options=["초급", "중급", "고급"], index=1)
    enable_explanations = st.sidebar.checkbox("해설 자동 생성", value=True)

    st.markdown("### 1) 강의 텍스트 업로드 or 붙여넣기")
    uploaded_file = st.file_uploader("강의 텍스트(.txt) 또는 오디오(.mp3/.wav) 업로드", type=["txt", "mp3", "wav", "m4a"], accept_multiple_files=False)

    raw_text_area = st.text_area("또는 강의 텍스트를 여기에 붙여넣기", height=200)

    # 우선 텍스트 우선 처리: 업로드된 파일이 txt면 읽고, 오디오면 transcribe 시도
    lecture_text = ""
    if uploaded_file is not None:
        fname = uploaded_file.name
        file_bytes = uploaded_file.read()
        if fname.lower().endswith(".txt"):
            lecture_text = file_bytes.decode("utf-8")
        else:
            # 오디오 파일 처리 시도
            st.info("오디오 파일이 업로드되었습니다. 음성->텍스트 변환(placeholder)을 시도합니다.")
            transcribed = transcribe_audio(file_bytes, fname)
            if transcribed:
                lecture_text = transcribed
            else:
                st.warning("오디오 자동 변환이 설정되지 않았습니다. 텍스트를 직접 붙여넣기 해주세요.")

    # 텍스트 영역 우선
    if raw_text_area.strip():
        lecture_text = raw_text_area

    lecture_text = clean_text(lecture_text)

    if not lecture_text:
        st.info("강의 텍스트가 없습니다. 텍스트를 붙여넣거나 .txt 파일을 업로드하세요.")
        return

    # 요약 수행
    if st.button("요약 및 문제 생성 시작"):
        with st.spinner("요약 생성 중..."):
            summary = summarize_text(lecture_text)
        if not summary:
            st.error("요약을 생성하지 못했습니다.")
            return

        st.markdown("## 요약")
        st.write(summary)

        # 문제 생성
        with st.spinner("문제 생성 중..."):
            mcqs = generate_mcq_from_summary(summary, n_questions=n_questions, difficulty=difficulty)

        if not mcqs:
            st.error("문제를 생성하지 못했습니다. 모델 출력 확인 필요")
            return

        # 해설 생성(선택)
        if enable_explanations:
            for item in mcqs:
                if not item.get("explanation"):
                    q = item.get("question", "")
                    choices = item.get("choices", [])
                    ans_idx = int(item.get("answer", 0))
                    item["explanation"] = generate_explanation(q, choices, ans_idx)

        # 시각화 - 간단한 카드 형태로 표시
        st.markdown("## 생성된 문제들")
        for i, item in enumerate(mcqs, 1):
            st.markdown(f"**{i}. {item.get('question')}**")
            choices = item.get('choices', [])
            for idx, c in enumerate(choices):
                prefix = "(정답)" if idx == int(item.get('answer', 0)) else ""
                st.write(f"{chr(65+idx)}. {c} {prefix}")
            if item.get('explanation'):
                with st.expander("해설 보기"):
                    st.write(item.get('explanation'))
            st.markdown("---")

        # JSON 다운로드
        result = {
            "summary": summary,
            "mcqs": mcqs,
            "meta": {"n_questions": n_questions, "difficulty": difficulty}
        }
        st.download_button("결과 JSON 다운로드", data=json.dumps(result, ensure_ascii=False, indent=2), file_name="lecture_mcqs.json", mime="application/json")

# ----------------------------------------------------------------------------
# 진입점
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
