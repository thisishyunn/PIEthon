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
import pandas as pd
from io import BytesIO
from fastapi.responses import StreamingResponse

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
            """당신은 FHIR 의료 데이터 분석 전문가입니다. 
다음 개인 의료 데이터를 분석하여 예방접종과 약물-식이 관리가 포함된 통합 건강 리포트를 작성해주세요.

=== 전체 FHIR 데이터 ===\n""" + combined
        )

        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": f"""
당신은 FHIR 의료 데이터 분석 전문가입니다. 
다음 개인 의료 데이터를 분석하여 예방접종과 약물-식이 관리가 포함된 통합 건강 리포트를 작성해주세요.
=== 분석 요구사항 ===

반드시 Plain Text 형식으로만 작성하고, 다음 구조로 리포트를 작성해주세요:

예방접종 분석 지침:
- 환자 데이터에서 추정 가능한 연령대, 성별을 고려해주세요
- 복용 약물로 추정되는 기저질환이 있다면 해당 질환별 고위험군 백신을 우선 권장해주세요
- 면역억제제나 항응고제 등 특별한 약물 복용시 접종 관련 주의사항을 명시해주세요
- 일반적인 권장사항보다는 개인의 위험도에 맞는 맞춤형 권장을 해주세요
- 불확실한 경우 "의료진과 상담 후 결정" 명시해주세요
- 복용 중인 약물별로 실제 상호작용이 확인된 음식만 안내해주세요
- 모든 약물에 공통 적용되는 일반적 주의사항은 피하고, 개별 약물의 특성에 맞는 정보만 제공해주세요
- 상호작용이 없는 약물의 경우 "특별한 식이 제한 없음"으로 명시해주세요
- 불확실한 정보보다는 확실한 정보만 제공해주세요
개인 통합 건강 관리 리포트
========================================

분석 기준일: {datetime.now().strftime('%Y년 %m월 %d일')}

----------------------------------------
1. 예방접종 현황 및 권장사항
----------------------------------------

[접종 완료된 백신]
- 백신명 | 접종일 | 유효기간

[권장 예방접종]
환자의 연령, 복용 약물, 추정 기저질환을 고려한 개인별 맞춤 권장사항:

[연령별 기본 권장 백신]
- 현재 연령대에서 권장되는 표준 예방접종
- 놓친 접종이 있다면 catch-up 일정

[기저질환별 고위험군 백신]
복용 약물로 추정되는 기저질환 기준:
- 당뇨병 환자: (해당시)
- 심혈관질환 환자: (해당시) 
- 면역저하 환자: (해당시)
- 만성폐질환 환자: (해당시)
- 기타 해당 질환별 권장 백신

[면역억제제 복용 고려사항]
복용 중인 약물이 면역에 영향을 주는 경우:
- 생백신 접종 금기 여부
- 접종 타이밍 조정 필요성
- 항체 형성률 고려사항

[계절성 백신]
- 독감 백신: 매년 접종 (고위험군 우선)
- 코로나19 백신: 부스터샷 일정 (위험도별)

[해외여행 대비]
- 여행 계획이 있다면 필요한 백신
- 지역별 풍토병 예방접종

[접종 일정 관리]
개인 상황을 고려한 접종 계획:
- 우선순위가 높은 백신: (위험도/시급성 기준)
- 다음 접종 권장 시기: (구체적 월/분기)
- 동시 접종 가능한 백신들:
- 접종 간격이 필요한 경우:

[접종 시 주의사항]
복용 중인 약물 고려:
- 면역억제제 복용시 주의사항
- 항응고제 복용시 주의사항 (근육주사 관련)
- 기타 약물별 특별 고려사항

----------------------------------------
2. 꾸준히 복용 중인 약물 분석
----------------------------------------

[장기 복용 약물 (3개월 이상)]
분석 기준: 지속적인 처방 패턴, 만료되지 않은 처방일수

약물 1: [약물명]
- 복용 기간: X개월
- 현재 상태: 복용 중 / 중단 추정
- 추정 질환: 
- 복용법: 

약물 2: [약물명]
- 복용 기간: X개월  
- 현재 상태: 복용 중 / 중단 추정
- 추정 질환:
- 복용법:

[단기 처방 약물]
- 항생제, 진통제 등 일시적 처방 내역

----------------------------------------
3. 약물별 식이 주의사항
----------------------------------------

각 장기 복용 약물에 대한 상세 가이드:

[약물명 1 - 식이 가이드]
이 약물과 실제로 상호작용하는 음식/음료만 분석해서 제공:

절대 피해야 할 음식/음료:
- (실제 상호작용이 있는 경우만 나열, 없으면 "특별한 제한 없음")

주의해서 섭취할 음식:
- (주의가 필요한 경우만 나열, 복용 간격 등 구체적 안내)

권장 복용 시간:
- 식전/식후/식간 구분 (약물 특성에 따라)
- 다른 약물과의 간격 (필요한 경우만)

[약물명 2 - 식이 가이드]
(동일한 형식으로 반복)

----------------------------------------
4. 종합 식단 관리 지침
----------------------------------------

[전체 약물 고려시 피해야 할 음식]
복용 중인 약물과 실제 상호작용이 확인된 음식/음료만 나열해주세요:
- 절대 금지 (심각한 상호작용): 
- 제한 필요 (주의 깊은 관리): 
- 시간 간격 필요 (흡수 방해): 

[일일 복약 스케줄]
아침 (X시): 
- 복용 약물명
- 식사 타이밍 (식전 30분/식후 1시간 등)

점심 (X시):
- 복용 약물명  
- 식사 타이밍

저녁 (X시):
- 복용 약물명
- 식사 타이밍

[영양제/건강식품 주의사항]
복용 중인 약물과 실제 상호작용 가능성이 있는 영양제만 안내:
- 상호작용 위험 영양제: (해당되는 경우만)
- 안전한 영양제 추천: (일반적으로 안전한 것들)
- 복용 간격 가이드: (필요한 경우만)

----------------------------------------
5. 건강 관리 권장사항
----------------------------------------

[정기 검진 권장]
- 현재 복용 약물 기준 필요한 정기 검사
- 부작용 모니터링 항목
- 검진 주기 권장

[생활 습관 개선]
- 약효 증진을 위한 생활 습관
- 부작용 예방을 위한 주의사항
- 운동/수면 권장사항

[응급상황 대비]
- 응급실 방문시 알려야 할 약물 정보
- 부작용 의심 증상
- 상비약 준비 권장사항

========================================

중요 안내사항:
- 이 분석은 정보 제공 목적이며 의학적 진단을 대체하지 않습니다
- 예방접종 및 약물 관련 모든 결정은 의료진과 상담 후 결정하세요  
- 약물 중단이나 변경은 반드시 의사와 상의하세요
- 응급상황시에는 즉시 의료기관을 방문하세요
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
            # 원본 리소스 저장(idempotent): 이미 존재하면 건너뜀
            existing = supabase.table("fhir_resources") \
                .select("id") \
                .eq("user_id", user_id) \
                .eq("fhir_id", fhir_id_val) \
                .execute()
            if existing.data:
                resource_id = existing.data[0]["id"]
            else:
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

@app.get("/export-data")
async def export_data():
    """PRO 응답과 EHR(예방접종, 투약, 청구) 데이터를 하나의 Excel 파일로 다운로드"""
    try:
        # PRO responses
        pro_resp = supabase.table("pro_responses").select("*").execute()
        pro_rows = pro_resp.data or []
        # PRO 응답의 hashed user_id 목록
        hashed_ids = list({r.get("user_id") for r in pro_rows if r.get("user_id")})
        # users 테이블에서 hashed name_hash -> 정수 id 매핑
        users_map_resp = supabase.table("users").select("id, name_hash").in_("name_hash", hashed_ids).execute()
        mapping = {u.get("name_hash"): u.get("id") for u in (users_map_resp.data or [])}
        # EHR 조회용 integer user_id 리스트
        ehr_user_ids = [mapping[h] for h in hashed_ids if mapping.get(h) is not None]

        # EHR tables
        immun_resp = supabase.table("immunizations").select("*").in_("user_id", ehr_user_ids).execute()
        meds_resp = supabase.table("medication_dispenses").select("*").in_("user_id", ehr_user_ids).execute()
        tc_resp   = supabase.table("treatment_claims").select("*").in_("user_id", ehr_user_ids).execute()
        immun_rows = immun_resp.data or []
        meds_rows  = meds_resp.data or []
        tc_rows    = tc_resp.data or []

        # DataFrame 생성
        df_pro   = pd.DataFrame(pro_rows)
        df_immun = pd.DataFrame(immun_rows)
        df_meds  = pd.DataFrame(meds_rows)
        df_tc    = pd.DataFrame(tc_rows)

        # Excel 파일 작성
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_pro.to_excel(writer, index=False, sheet_name="PRO Responses")
            df_immun.to_excel(writer, index=False, sheet_name="Immunizations")
            df_meds.to_excel(writer, index=False, sheet_name="MedicationDispenses")
            df_tc.to_excel(writer, index=False, sheet_name="TreatmentClaims")
        buf.seek(0)
        headers = {"Content-Disposition": "attachment; filename=export.xlsx"}
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 