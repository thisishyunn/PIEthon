from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import io
from PyPDF2 import PdfReader
import os
import glob
from openai import OpenAI
import json
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

load_dotenv()
app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:3000",  # 로컬 프론트엔드
    "https://your-frontend-domain.com",  # 배포된 프론트엔드 도메인
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

supabase: Client = create_client(supabase_url, supabase_key)

@app.get("/ping")
async def ping():
    return {"message": "pong"}

@app.post("/extract")
async def extract_pdf(file: UploadFile = File(...), password: str = Form(...)):
    try:
        content = await file.read()
        reader = PdfReader(io.BytesIO(content))
        if reader.is_encrypted:
            result = reader.decrypt(password)
            if result == 0:
                raise HTTPException(status_code=400, detail="비밀번호가 올바르지 않습니다.")
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return {"content": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 처리 중 오류 발생: {str(e)}")

@app.post("/health-report")
async def health_report(files: List[UploadFile] = File(default=None)):
    """FHIR JSON 파일을 직접 업로드하거나(여러 개 가능) 
    업로드가 없으면 서버의 fhir/ 디렉터리를 읽어 GPT 건강 보고서를 생성한다."""

    try:
        combined = ""

        # 1) 요청으로부터 업로드된 파일이 있는 경우 우선 사용
        if files:
            for uf in files:
                content_bytes = await uf.read()
                try:
                    combined += content_bytes.decode("utf-8") + "\n"
                except UnicodeDecodeError:
                    raise HTTPException(status_code=400, detail=f"파일 {uf.filename} 은(는) UTF-8 인코딩된 JSON이 아닙니다.")

        # 2) 업로드가 없으면 기존 fhir 디렉터리의 파일 사용
        if not combined:
            fhir_dir = os.path.join(os.path.dirname(__file__), "fhir")
            local_files = glob.glob(os.path.join(fhir_dir, "*"))
            if not local_files:
                raise HTTPException(status_code=404, detail="FHIR 데이터 파일을 찾을 수 없습니다.")
            for file_path in local_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    combined += f.read() + "\n"

        # GPT 프롬프트 구성 및 호출
        prompt = (
            "다음 환자의 FHIR 데이터를 바탕으로 종합적인 건강 보고서를 작성해주세요\n" + combined
        )

        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": f"""
        당신은 FHIR 의료 데이터 분석 전문가입니다. 
        Plain-Text 형식 리포트를 작성해야 하며 의학적 진단을 대체하지 않는다는
        고지를 항상 포함합니다. 다음 구조로 리포트를 작성해주세요:

        # ========================================
        개인 건강 분석 리포트

        분석 기준일: {datetime.now().strftime('%Y년 %m월 %d일')}

        ---

        1. 약물 현황 요약

        ---

        - 현재 복용 추정 약물: X개 (최근 30일 이내 처방 기준)
        - 중단된 약물: X개
        - 상태 불분명 약물: X개
        - 전체 투약 기록: X건

        ---

        1. 약물 상태 분류

        ---

        [현재 복용 중인 약물]

        - 약물명 (성분명) | 처방일 | 복용법

        [중단된 약물]

        - 약물명 (성분명) | 마지막 처방일 | 중단 추정일

        [일시적 처방 약물]

        - 약물명 | 처방 기간 | 용도 추정

        ---

        1. 시간별 약물 변화

        ---

        - 최초 처방: YYYY-MM
        - 최근 처방: YYYY-MM
        - 약물 추가 이력: (시기별로 정리)
        - 약물 중단 이력: (시기별로 정리)

        ---

        1. 건강 상태 추정

        ---

        - 투약 기록으로 추정되는 질환/증상
        - 만성 질환 관리 현황
        - 급성 치료 이력

        ---

        1. 의료 이용 패턴

        ---

        - 주요 이용 의료기관/약국
        - 처방 주기 패턴
        - 계절별/시기별 특이사항

        ---

        1. 주의사항 및 권장사항

        ---

        - 잠재적 약물 상호작용
        - 복약 관리 개선점
        - 의료진 상담 권장 사항

        ========================================

        중요사항:

        - 이 분석은 정보 제공 목적이며 의학적 진단을 대체하지 않습니다
        - 모든 의약품 관련 결정은 의료진과 상담 후 결정하세요
        - 응급상황 시에는 즉시 의료기관을 방문하세요
"""},
                {"role": "user", "content": prompt},
            ],
        )

        report = response.choices[0].message.content
        return {"report": report}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"건강 보고서 생성 중 오류 발생: {str(e)}"
        )

class FHIRIngestRequest(BaseModel):
    name_hash: str
    full_name: Optional[str] = None
    resource: dict

@app.post("/ingest-fhir")
async def ingest_fhir(request: FHIRIngestRequest):
    try:
        # 1) 사용자 조회 또는 생성
        user_resp = supabase.table("users").select("id").eq("name_hash", request.name_hash).execute()
        if user_resp.data:
            user_id = user_resp.data[0]["id"]
        else:
            insert_user = supabase.table("users").insert({"name_hash": request.name_hash, "full_name": request.full_name}).execute()
            user_id = insert_user.data[0]["id"]

        # 2) 원본 FHIR 리소스 저장 및 매핑 (publicData 처리)
        raw = request.resource
        # publicData 배열이 있으면 여러 리소스 처리, 없으면 단일 리소스로 처리
        resources = []
        if isinstance(raw.get("publicData"), list):
            for item in raw["publicData"]:
                if isinstance(item, dict) and "resource" in item:
                    resources.append(item["resource"])
        else:
            resources.append(raw)

        results = []
        for res in resources:
            resource_type = res.get("resourceType")
            fhir_id_val = res.get("id")
            # 원본 리소스 저장
            insert_fhir = supabase.table("fhir_resources").insert({
                "user_id": user_id,
                "resource_type": resource_type,
                "fhir_id": fhir_id_val,
                "data": res
            }).execute()
            resource_id = insert_fhir.data[0]["id"]

            # 리소스 타입별 매핑
            if resource_type == "MedicationDispense":
                # medication 정보
                med_ref = res.get("medicationReference", {})
                med_res = med_ref.get("resource", {})
                coding = med_res.get("code", {}).get("coding", [{}])[0]
                medication_code = coding.get("code")
                print("medication_code", medication_code)
                medication_name = coding.get("display")
                print("medication_name", medication_name)
                # pharmacy name 추출 (performer 배열의 actor.resource.name)
                pharmacy_name = None
                perf = res.get("performer", [])
                if isinstance(perf, list) and perf:
                    actor = perf[0].get("actor", {})
                    org = actor.get("resource", {})
                    pharmacy_name = org.get("name")
                print("pharmacy_name", pharmacy_name)
                when_prepared = res.get("whenPrepared", "").split("T")[0]
                print("when_prepared", when_prepared)
                days_supply = res.get("daysSupply", {}).get("value")

                print({
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "medication_code": medication_code,
                    "medication_name": medication_name,
                    "pharmacy_name": pharmacy_name,
                    "when_prepared": when_prepared,
                    "days_supply": days_supply,
                })

                insert_md = supabase.table("medication_dispenses").insert({
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "medication_code": medication_code,
                    "medication_name": medication_name,
                    "pharmacy_name": pharmacy_name,
                    "when_prepared": when_prepared,
                    "days_supply": days_supply,
                }).execute()
            elif resource_type == "ExplanationOfBenefit":
                claim_type_coding = res.get("type", {}).get("coding", [{}])[0]
                claim_type = claim_type_coding.get("code") or claim_type_coding.get("display")
                # created_date: billablePeriod.start 우선 사용
                billable = res.get("billablePeriod", {})
                created_date = billable.get("start", res.get("created", "")).split("T")[0]
                copay_amount = None
                benefit_amount = None
                totals = res.get("total", [])
                if isinstance(totals, list):
                    for t in totals:
                        cat = t.get("category", {}).get("coding", [{}])[0]
                        code_key = cat.get("code")
                        val = t.get("amount", {}).get("value")
                        if code_key == "copay":
                            copay_amount = val
                        elif code_key == "benefit":
                            benefit_amount = val
                insert_tc = supabase.table("treatment_claims").insert({
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "claim_type": claim_type,
                    "created_date": created_date,
                    "copay_amount": copay_amount,
                    "benefit_amount": benefit_amount
                }).execute()
            elif resource_type == "Immunization":
                vc = res.get("vaccineCode", {}).get("coding", [{}])[0]
                vaccine_name = vc.get("display")
                occurrence_date = res.get("occurrenceDateTime", "").split("T")[0]
                # dose number 체크
                dose_number = res.get("doseNumber")
                if not dose_number:
                    prot = res.get("protocolApplied", [])
                    if isinstance(prot, list) and prot:
                        dose_number = prot[0].get("doseNumberPositiveInt") or prot[0].get("doseNumber")

                # performer name
                perf_name = None
                p_list = res.get("performer", [])
                if isinstance(p_list, list) and p_list:
                    actor = p_list[0].get("actor", {})
                    org = actor.get("resource", {})
                    perf_name = org.get("name")

                insert_im = supabase.table("immunizations").insert({
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "vaccine_name": vaccine_name,
                    "occurrence_date": occurrence_date,
                    "dose_number": dose_number,
                    "performer_name": perf_name
                }).execute()
            else:
                # 지원하지 않는 리소스 타입은 건너뜀
                continue

            results.append({"resource_type": resource_type, "resource_id": resource_id})

        return {"status": "success", "user_id": user_id, "results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/debug/clear-data")
async def clear_data():
    """디버깅용: users 테이블을 제외한 모든 데이터 삭제"""
    try:
        tables = [
            "medication_dispenses",
            "treatment_claims",
            "immunizations",
            "fhir_resources",
        ]
        results = {}
        for tbl in tables:
            res = supabase.table(tbl).delete().neq("id", 0).execute()
            results[tbl] = len(res.data) if res.data else 0
        return {"status": "cleared", "deleted_rows": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"데이터 삭제 오류: {str(e)}")

# PROResponse 모델 및 /pro-responses 엔드포인트 추가
class PROResponse(BaseModel):
    user_id: str
    response: dict

@app.post("/pro-responses")
async def create_pro_response(request: PROResponse):
    """PRO 설문 응답을 받아 pro_responses 테이블에 저장"""
    try:
        insert_res = supabase.table("pro_responses").insert({
            "user_id": request.user_id,
            "response": request.response
        }).execute()
        pro_id = insert_res.data[0]["id"]
        return {"status": "success", "id": pro_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PRO 응답 저장 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 