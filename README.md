# MavenAIProject

This project contains the bootcamp-style Streamlit frontend and FastAPI backend
as uv workspace packages, plus the original learning materials under `src/` and
`notebooks/`.

## Run with Docker Compose

1. Copy `.env.example` to `.env` and add at least one provider API key.
2. From the repository root, build and start the services:

   ```powershell
   docker compose up --build
   ```

3. Open the Streamlit UI at <http://localhost:8501>. The FastAPI docs are at
   <http://localhost:8000/docs>.

Stop the services with `Ctrl+C`, or run `docker compose down` in another
terminal.