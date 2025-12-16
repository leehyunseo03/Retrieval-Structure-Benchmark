import os
import json
import requests
import re
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

# =========================================================
# [설정 영역]
# =========================================================
load_dotenv()
BASE_DIR = os.environ.get('FinanceBench_DIR')
FB_DATA_DIR = os.path.join(BASE_DIR, "financebench")
PDF_DIR = os.path.join(BASE_DIR, "financebench_pdfs")
JSON_SAVE_PATH = os.path.join(FB_DATA_DIR, "financebench_data.json")

# 다운로드할 PDF 개수 제한 (None이면 전체 - 약 100개 내외지만 파일이 큼)
LIMIT_PDF_DOWNLOAD = 30 

# =========================================================
# [기능 구현]
# =========================================================

def setup_directories():
    if not os.path.exists(FB_DATA_DIR):
        os.makedirs(FB_DATA_DIR)
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
    print(f"📂 디렉토리 확인:\n - JSON: {FB_DATA_DIR}\n - PDF: {PDF_DIR}")

def sanitize_filename(name):
    """파일명으로 쓸 수 없는 문자 제거 및 공백 처리"""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name.replace(" ", "_")

def download_and_process_data():
    print("⬇️  Hugging Face에서 FinanceBench 데이터셋 로드 중...")
    try:
        # FinanceBench는 보통 'train' 스플릿 하나만 존재함
        ds = load_dataset("PatronusAI/financebench", split="train")
    except Exception as e:
        print(f"❌ 데이터셋 로드 실패: {e}")
        return {}

    formatted_data = {}
    unique_docs = {} # PDF 다운로드를 위한 중복 제거용 딕셔너리 {doc_name: doc_link}

    print("🔄 데이터 변환 및 PDF 링크 추출 중...")
    for entry in ds:
        # FinanceBench의 고유 식별자는 financebench_id
        qid = entry.get('financebench_id')
        doc_name = entry.get('doc_name')
        doc_link = entry.get('doc_link')
        
        # 파일명 안전하게 변환
        safe_doc_name = sanitize_filename(doc_name)
        
        # 다운로드 목록에 추가
        if safe_doc_name not in unique_docs and doc_link:
            unique_docs[safe_doc_name] = doc_link

        # 평가용 데이터 구조 생성
        # 구조: { "DOC_NAME": { "questions": [...] } } 형태로 저장 (문서 중심)
        if safe_doc_name not in formatted_data:
            formatted_data[safe_doc_name] = {
                "original_doc_name": doc_name,
                "doc_link": doc_link,
                "qas": []
            }
        
        formatted_data[safe_doc_name]["qas"].append({
            "qid": qid,
            "question": entry.get('question'),
            "answer": entry.get('answer'), # 정답 텍스트
            "evidence_text": entry.get('evidence_text'), # 근거 문장
            "page_number": entry.get('page_number')
        })

    # JSON 저장
    with open(JSON_SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ JSON 데이터 저장 완료: {JSON_SAVE_PATH}")
    return unique_docs

def download_pdfs(doc_map):
    print(f"\n⬇️  PDF 다운로드 시작 (총 {len(doc_map)}개 문서 중 {LIMIT_PDF_DOWNLOAD if LIMIT_PDF_DOWNLOAD else '전체'} 대상)")
    
    target_docs = list(doc_map.items())
    if LIMIT_PDF_DOWNLOAD:
        target_docs = target_docs[:LIMIT_PDF_DOWNLOAD]
    
    success = 0
    fail = 0
    skipped = 0

    for doc_name, link in tqdm(target_docs, desc="Downloading PDFs"):
        save_path = os.path.join(PDF_DIR, f"{doc_name}.pdf")
        
        if os.path.exists(save_path):
            skipped += 1
            continue
            
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(link, headers=headers, timeout=60, stream=True)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                success += 1
            else:
                # print(f"Failed {doc_name}: Status {response.status_code}")
                fail += 1
        except Exception as e:
            # print(f"Error {doc_name}: {e}")
            fail += 1
            
    print("\n" + "="*40)
    print(f"🎉 다운로드 결과")
    print(f" - 성공: {success}")
    print(f" - 실패: {fail}")
    print(f" - 스킵(이미 있음): {skipped}")
    print(f" - 저장 경로: {PDF_DIR}")
    print("="*40)

if __name__ == "__main__":
    setup_directories()
    
    # 1. 데이터 처리 및 링크 추출
    docs_to_download = download_and_process_data()
    
    # 2. PDF 다운로드
    if docs_to_download:
        download_pdfs(docs_to_download)