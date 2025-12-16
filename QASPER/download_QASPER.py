import os
import json
import requests
import time
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm

# =========================================================
# [설정 영역]
# =========================================================
load_dotenv()
# 데이터를 저장할 기본 경로
BASE_DIR = os.environ.get('QASPER_DIR') 

# 저장될 경로 설정
QASPER_DATA_DIR = os.path.join(BASE_DIR, "qasper")
PDF_DIR = os.path.join(BASE_DIR, "qasper_pdfs")
JSON_SAVE_PATH = os.path.join(QASPER_DATA_DIR, "qasper-dev-v0.3.json")

# PDF 다운로드 개수 제한 (None이면 전체 다운로드)
LIMIT_PDF_DOWNLOAD = 50

# =========================================================
# [기능 구현]
# =========================================================

def setup_directories():
    if not os.path.exists(QASPER_DATA_DIR):
        os.makedirs(QASPER_DATA_DIR)
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    print(f"📂 디렉토리 확인 완료:\n - JSON: {QASPER_DATA_DIR}\n - PDF: {PDF_DIR}")

def download_and_convert_json():
    print("⬇️  Hugging Face에서 QASPER 데이터셋(validation) 다운로드 중...")
    # 'validation' 셋을 dev 셋으로 사용합니다.
    ds = load_dataset("allenai/qasper", split="validation")
    
    formatted_data = {}
    
    print("🔄 데이터 변환 중 (HF format -> Original Dictionary format)...")
    for entry in ds:
        paper_id = entry['id']
        
        # Hugging Face 포맷을 이전 평가 코드가 읽을 수 있는 Dict 형태로 변환
        formatted_data[paper_id] = {
            "title": entry['title'],
            "abstract": entry['abstract'],
            "qas": []
        }
        
        # QA 데이터 구조 변환
        # HF 데이터셋의 'qas'는 리스트들의 딕셔너리 형태일 수 있어 파싱 필요
        qas_raw = entry['qas']
        
        # question_id와 question 개수만큼 순회
        num_qas = len(qas_raw['question_id'])
        
        for i in range(num_qas):
            qa_obj = {
                "question_id": qas_raw['question_id'][i],
                "question": qas_raw['question'][i],
                "answers": []
            }
            
            # Answers 처리 (answer 리스트 안의 구조 처리)
            # HF Dataset의 구조가 복잡하므로, 단순화하여 정답 텍스트가 있는 부분만 추출 시도
            # (여기서는 기본적인 구조만 잡고, 실제 내용은 원본 데이터 구조를 최대한 따름)
            
            raw_answers = qas_raw['answers'][i] # This is a dict with lists
            
            # answer_id 개수만큼 순회 (하나의 질문에 여러 답변자가 있을 수 있음)
            # HF QASPER 구조: answers -> {'answer': [{'free_form_answer': ..., 'highlighted_evidence': ...}]}
            
            # 데이터셋 버전/구조에 따라 다를 수 있으므로 안전하게 처리
            answer_list = raw_answers.get('answer', [])
            
            converted_answers = []
            for ans in answer_list:
                converted_answers.append({"answer": ans})
                
            qa_obj["answers"] = converted_answers
            formatted_data[paper_id]["qas"].append(qa_obj)

    # JSON 파일로 저장
    with open(JSON_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON 데이터 저장 완료: {JSON_SAVE_PATH} (총 {len(formatted_data)}개 논문)")
    return list(formatted_data.keys())

def download_pdfs(paper_ids):
    print(f"\n⬇️  PDF 다운로드 시작 (총 {len(paper_ids)}개 대상)")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 제한 설정 적용
    target_ids = paper_ids[:LIMIT_PDF_DOWNLOAD] if LIMIT_PDF_DOWNLOAD else paper_ids
    
    for pid in tqdm(target_ids, desc="Downloading PDFs"):
        # arXiv ID로 PDF URL 생성
        pdf_url = f"https://arxiv.org/pdf/{pid}.pdf"
        save_path = os.path.join(PDF_DIR, f"{pid}.pdf")
        
        # 이미 존재하면 스킵
        if os.path.exists(save_path):
            skip_count += 1
            continue
            
        try:
            # arXiv 서버 부하 방지를 위한 딜레이 (필수)
            time.sleep(3) 
            
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
            else:
                # print(f"Failed to download {pid}: Status {response.status_code}")
                fail_count += 1
        except Exception as e:
            # print(f"Error downloading {pid}: {e}")
            fail_count += 1
            
    print("\n" + "="*40)
    print(f"🎉 다운로드 요약")
    print(f" - 성공: {success_count}")
    print(f" - 실패: {fail_count}")
    print(f" - 스킵(이미 있음): {skip_count}")
    print(f" - 저장 경로: {PDF_DIR}")
    print("="*40)

if __name__ == "__main__":
    setup_directories()
    
    # 1. JSON 데이터 준비
    all_paper_ids = download_and_convert_json()
    
    # 2. PDF 다운로드
    download_pdfs(all_paper_ids)