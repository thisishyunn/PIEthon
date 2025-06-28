import requests

def extract_pdf(file_path, password, url='http://localhost:8000/extract'):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {'password': password}
        response = requests.post(url, files=files, data=data)
    try:
        response.raise_for_status()
        json_data = response.json()
        print('추출된 내용:\n', json_data.get('content', ''))
    except requests.exceptions.HTTPError as err:
        print(f'HTTP error: {err}')
        print('응답 바디:', response.text)
    except ValueError:
        print('JSON 파싱 실패, 응답:', response.text)


if __name__ == '__main__':
    # 여기에 PDF 파일 경로와 비밀번호를 입력하세요.
    PDF_PATH = 'pbh2.pdf'
    PASSWORD = '0000'
    extract_pdf(PDF_PATH, PASSWORD) 