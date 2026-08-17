# Solar AI Pakistan Backend

Small FastAPI API for the website chat.

## Run

```sh
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Endpoints:
- `GET /health`
- `POST /chat` with `{ "message": "..." }`
