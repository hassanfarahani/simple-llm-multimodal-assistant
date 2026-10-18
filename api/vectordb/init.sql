-- to be able to rerun this file, you first need to remove the named volume in compose file (docker compose down -v)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(3072) -- Adjust dimension to match your model
);

