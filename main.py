from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import io
from PyPDF2 import PdfReader
import os
import glob
from openai import OpenAI
import json
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from supabase import create_client, Client
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/health-report")
async def health_report():
    try:
        fhir_dir = os.path.join(os.path.dirname(__file__), "fhir")
        files = glob.glob(os.path.join(fhir_dir, "*"))
        if not files:
            raise HTTPException(status_code=404, detail="FHIR 데이터 파일을 찾을 수 없습니다.")
        combined = ""
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                combined += f.read() + "\n"
        prompt = f"다음 환자의 FHIR 데이터를 바탕으로 종합적인 건강 보고서를 작성해줘:\n{combined}"
        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "당신은 의료 보고서 생성 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
        )
        report = response.choices[0].message.content
        print(report)
        return {"report": report}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"건강 보고서 생성 중 오류 발생: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 