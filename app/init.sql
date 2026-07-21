-- Runs once on first Postgres init (empty data dir). LangGraph memory tables are
-- created separately by saver.setup() at bot startup.

CREATE TABLE IF NOT EXISTS tenants (
    tenant_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    is_active   BOOLEAN NOT NULL DEFAULT true,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chats (
    chat_id     BIGINT PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES tenants(tenant_id),
    is_active   BOOLEAN NOT NULL DEFAULT true,
    note        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS seen_updates (
    update_id   BIGINT PRIMARY KEY,
    seen_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tenant_roles (
    tenant_id   TEXT NOT NULL REFERENCES tenants(tenant_id),
    user_id     BIGINT NOT NULL,
    role        TEXT NOT NULL CHECK (role = 'admin'),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, user_id)
);

CREATE TABLE IF NOT EXISTS documents (
    tenant_id    TEXT NOT NULL REFERENCES tenants(tenant_id),
    source       TEXT NOT NULL,
    uploaded_by  BIGINT,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    chunks       INT,
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
    error        TEXT,
    PRIMARY KEY (tenant_id, source)
);

INSERT INTO tenants (tenant_id, name) VALUES ('default', 'Default')
    ON CONFLICT DO NOTHING;

-- No admin is seeded here: admin rights are granted out-of-band by the operator (there is
-- no in-bot privilege escalation).
