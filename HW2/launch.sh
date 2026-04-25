cd backend || exit 1
(
sleep 1
# exec sh -c 'open http://localhost:8000'
open http://localhost:8000
) &
exec uvicorn main:app --reload --reload-dir "$PWD/.." --port 8000 --host 127.0.0.1

# kill -9 $(lsof -t -i:8000)
