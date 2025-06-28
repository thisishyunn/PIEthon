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
import re

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
            """
당신은 FHIR 의료 데이터 분석 전문가입니다. 
다음 개인 의료 데이터를 분석하여 흥미로운 건강 관리 프로필을 작성해주세요.

=== 전체 FHIR 데이터 ===\n""" + combined
        )

        response = client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": f"""
=== 분석 요구사항 ===

반드시 Plain Text 형식으로만 작성하고, 다음 구조로 리포트를 작성해주세요.
각 카테고리별로 제시된 예시 중에서 데이터에 가장 적합한 하나만 선택해서 출력해주세요:

========================================
나의 건강 관리 프로필
========================================

분석 기준일: {datetime.now().strftime('%Y년 %m월 %d일')}

----------------------------------------
📊 가장 자주 복용한 약물 TOP 3
----------------------------------------
처방 횟수와 총 복용 기간을 종합하여 순위를 매겨주세요:
1위: [약물명] - 총 X회 처방, X일간 복용
2위: [약물명] - 총 X회 처방, X일간 복용  
3위: [약물명] - 총 X회 처방, X일간 복용

----------------------------------------
🏥 나의 의료 이용 스타일
----------------------------------------
다음 중 하나만 선택해서 출력:

A) "한 병원 충성형"
주로 1-2개 병원만 이용하는 안정파입니다. 
신뢰하는 의료진과 꾸준한 관계를 유지하는 스타일이군요!

B) "이곳저곳 탐험형"  
다양한 의료기관을 경험해보는 탐험가입니다.
여러 전문의의 의견을 듣고 비교하는 것을 선호하시는군요!

C) "계절 민감형"
특정 계절에 처방이 몰리는 패턴을 보입니다.
환절기나 특정 시기에 건강 관리가 집중되는 타입이네요!

D) "연중 꾸준형"
1년 내내 꾸준히 관리하는 타입입니다.
계절에 관계없이 일정한 주기로 건강을 챙기시는군요!

----------------------------------------
💊 나의 약물 관리 특징  
----------------------------------------
다음 중 하나만 선택해서 출력:

A) "장기 동반자형"
6개월 이상 꾸준히 복용하는 약물이 많습니다.
건강을 장기적 관점에서 체계적으로 관리하고 계시네요!

B) "단기 해결형"
필요할 때만 짧게 복용하는 스타일입니다.
문제가 생겼을 때 빠르게 해결하고 정리하는 타입이군요!

C) "혼합 관리형"
장기약과 단기약을 적절히 조합하여 사용합니다.
상황에 맞게 유연하게 대응하는 균형잡힌 스타일이네요!

D) "규칙적 관리형"
일정한 주기로 처방받는 패턴을 보입니다.
계획적이고 체계적인 건강 관리를 실천하고 계시군요!

E) "비정기 방문형"
필요에 따라 불규칙하게 처방받는 패턴입니다.
자신의 몸 상태를 잘 파악하고 필요시에만 대응하는 타입이네요!

----------------------------------------
📈 나의 건강 변화 스토리
----------------------------------------
다음 중 하나만 선택해서 출력:

A) "건강 안정기"
약물 종류가 일정하게 유지되고 있습니다.
현재 상태를 잘 유지하며 안정적으로 관리하고 계시는군요!

B) "관리 확대기"  
시간이 지나면서 약물이 점진적으로 증가하고 있습니다.
건강 관리 영역이 넓어지며 더 세심한 케어를 받고 계시네요!

C) "건강 개선기"
최근 약물이 줄어드는 긍정적 변화를 보입니다.
꾸준한 관리의 효과가 나타나고 있는 것 같아요!

D) "초기 집중형"
처음에 많이 처방받고 이후 안정화된 패턴입니다.
초기 문제를 집중적으로 해결한 후 관리 모드로 전환하셨군요!

E) "점진적 증가형"
시간이 지나면서 의료 이용이 늘어나고 있습니다.
나이나 환경 변화에 맞춰 건강 관리를 강화하고 계시네요!

F) "주기적 파동형"
일정 주기로 처방이 늘었다 줄었다 반복됩니다.
계절적 요인이나 생활 패턴의 영향을 받는 타입이군요!

----------------------------------------
🎯 종합 건강 관리 특징
----------------------------------------
다음 중 하나만 선택해서 출력:

A) "체계적 관리자"
꾸준하고 계획적으로 건강을 관리하는 타입입니다. 
신뢰하는 의료진과 장기적 관계를 유지하며, 건강 변화에 체계적으로 대응하고 계시네요!

B) "필요시 대응형"
문제가 생겼을 때 빠르게 해결하는 타입입니다.
평소에는 큰 신경 쓰지 않다가도 필요할 때는 집중적으로 관리하시는군요!

C) "점진적 적응형"  
변화에 천천히 적응해가는 신중한 타입입니다.
새로운 치료나 약물에 대해 충분히 고민하고 단계적으로 접근하시는군요!

D) "적극적 개선형"
건강 개선을 위해 다양한 시도를 하는 타입입니다.
여러 방법을 시도해보며 자신에게 맞는 최적의 관리법을 찾아가시는군요!

E) "안정 추구형"
익숙하고 신뢰할 수 있는 방식을 선호합니다.
검증된 방법으로 꾸준히 관리하며 급격한 변화보다는 안정성을 중시하시네요!

F) "균형 조절형"
적당한 선에서 건강과 일상의 균형을 맞춥니다.
과도하지 않으면서도 필요한 만큼은 챙기는 현실적인 접근을 하시는군요!

G) "문제 해결형"
발생한 문제를 집중적으로 해결하는 스타일입니다.
이슈가 생기면 철저히 파악하고 해결할 때까지 집중하는 타입이군요!

H) "예방 중심형"
미리미리 대비하는 것을 중요하게 생각합니다.
문제가 생기기 전에 예방하고 관리하는 것을 선호하시는군요!

----------------------------------------
💊 주요 약물 식이 가이드
----------------------------------------
장기 복용 약물 중심으로 실제 상호작용이 있는 경우만 안내:

[약물명]: 
- 피해야 할 음식/음료: (있는 경우만, 없으면 "특별한 제한 없음")
- 권장 복용 시간: 식전/식후/식간
- 주의사항: (필요한 경우만)

----------------------------------------
🏥 예방접종 권장사항  
----------------------------------------
복용 약물과 추정 연령을 고려한 맞춤 권장:

[우선 권장 백신]
- 기저질환 고려시: (당뇨, 심혈관질환 등 해당시)
- 연령대 권장: (해당 연령대 표준 백신)
- 계절 백신: 독감, 코로나19 등

[접종 시 주의사항]
- 복용 약물 관련: (면역억제제, 항응고제 등 해당시만)

========================================

중요 안내사항:
- 이 분석은 흥미로운 패턴 파악을 위한 정보 제공 목적입니다
- 의학적 결정은 반드시 의료진과 상담 후 하세요
- 약물 변경이나 중단은 의사 지시에 따라서만 하세요

분석 지침:
- 각 카테고리별로 제시된 예시 중 데이터에 가장 적합한 하나만 선택해주세요
- 복용 중인 약물별로 실제 상호작용이 확인된 음식만 안내해주세요  
- 상호작용이 없는 약물의 경우 "특별한 식이 제한 없음"으로 명시해주세요
- 예방접종은 복용 약물로 추정되는 기저질환과 연령을 고려해 개인 맞춤형으로 권장해주세요
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

                # 나이 계산 (주민등록번호 앞 7자리 이용)
                age = None
                subj = res.get("subject", {})
                pat = subj.get("resource", {})
                ident_list = pat.get("identifier", [])
                rrn_val = None
                for ident in ident_list:
                    if ident.get("system") == "http://mois.go.kr/rnn":
                        rrn_val = ident.get("value", "")
                        break
                if rrn_val:
                    digits = re.sub(r"\D", "", rrn_val)
                    if len(digits) >= 7:
                        yy = int(digits[0:2])
                        mm = int(digits[2:4])
                        dd = int(digits[4:6])
                        code = digits[6]
                        year = 1900 + yy if code in ("1", "2") else 2000 + yy
                        dob = datetime(year, mm, dd).date()
                        prep_date = datetime.fromisoformat(when_prepared).date()
                        age = prep_date.year - dob.year - ((prep_date.month, prep_date.day) < (dob.month, dob.day))

                print({
                    "user_id": user_id,
                    "resource_id": resource_id,
                    "medication_code": medication_code,
                    "medication_name": medication_name,
                    "pharmacy_name": pharmacy_name,
                    "when_prepared": when_prepared,
                    "days_supply": days_supply,
                    "age": age,
                })

                # idempotent: 이미 삽입된 매핑은 건너뜀
                md_exists = supabase.table("medication_dispenses") \
                    .select("id") \
                    .eq("resource_id", resource_id) \
                    .execute()
                if not md_exists.data:
                    supabase.table("medication_dispenses").insert({
                        "user_id": user_id,
                        "resource_id": resource_id,
                        "medication_code": medication_code,
                        "medication_name": medication_name,
                        "pharmacy_name": pharmacy_name,
                        "when_prepared": when_prepared,
                        "days_supply": days_supply,
                        "age": age,
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
                # idempotent: 이미 삽입된 매핑은 건너뜀
                tc_exists = supabase.table("treatment_claims") \
                    .select("id") \
                    .eq("resource_id", resource_id) \
                    .execute()
                if not tc_exists.data:
                    supabase.table("treatment_claims").insert({
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

                # idempotent: 이미 삽입된 매핑은 건너뜀
                im_exists = supabase.table("immunizations") \
                    .select("id") \
                    .eq("resource_id", resource_id) \
                    .execute()
                if not im_exists.data:
                    supabase.table("immunizations").insert({
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