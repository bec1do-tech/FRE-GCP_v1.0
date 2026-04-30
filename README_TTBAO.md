## Things to be aware of(TTBAO):

# 1. px proxy (run once, keep in background)
& "C:\Users\BEC1DO\Desktop\canbek\FRE_GCP_v1.0\.venv\Scripts\px.exe" --proxy=rb-proxy-de.bosch.com:8080 --port=3128

# 2. ADK (new terminal, with proxy vars)
$env:HTTPS_PROXY="http://127.0.0.1:3128"; $env:HTTP_PROXY="http://127.0.0.1:3128"; $env:NO_PROXY="localhost,127.0.0.1"
Set-Location "C:\Users\BEC1DO\Desktop\canbek\FRE_GCP_v1.0"
& ".venv\Scripts\adk.exe" web

# 2b. ADK with PERSISTENT SESSIONS (survives browser refresh / ADK restart)
#     Stores session state in the same PostgreSQL used by the search system.
#     Requires docker compose to be running (postgres on localhost:5432).
$env:HTTPS_PROXY="http://127.0.0.1:3128"; $env:HTTP_PROXY="http://127.0.0.1:3128"; $env:NO_PROXY="localhost,127.0.0.1"
Set-Location "C:\Users\BEC1DO\Desktop\canbek\FRE_GCP_v1.0"
$pg = (Get-Content .env | Select-String "^POSTGRES_PASSWORD=").ToString().Split("=",2)[1]
& ".venv\Scripts\adk.exe" web --session_db_url "postgresql+psycopg2://postgres:$pg@localhost:5432/cognitive_search"