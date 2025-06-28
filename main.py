from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import io
from PyPDF2 import PdfReader
import os
import glob
from openai import OpenAI
client = OpenAI(api_key="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")

app = FastAPI()

# openai.api_key = os.getenv("OPENAI_API_KEY")

# client.api_key = "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"

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 