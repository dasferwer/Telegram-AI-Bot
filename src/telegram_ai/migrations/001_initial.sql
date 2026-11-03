CREATE TABLE conversations (
    bot_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    epoch BIGINT NOT NULL DEFAULT 0,
    last_request TIMESTAMPTZ,
    PRIMARY KEY (bot_id, user_id)
);

CREATE TABLE jobs (
    bot_id BIGINT NOT NULL,
    update_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    epoch BIGINT NOT NULL,
    question TEXT,
    answer TEXT,
    is_turn BOOLEAN NOT NULL DEFAULT FALSE,
    state TEXT NOT NULL DEFAULT 'pending'
        CHECK (state IN ('pending', 'running', 'done', 'cancelled', 'failed')),
    generation_attempts INTEGER NOT NULL DEFAULT 0,
    send_attempts INTEGER NOT NULL DEFAULT 0,
    sent_chunks INTEGER NOT NULL DEFAULT 0,
    available_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_until TIMESTAMPTZ,
    owner UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    PRIMARY KEY (bot_id, update_id),
    FOREIGN KEY (bot_id, user_id) REFERENCES conversations(bot_id, user_id)
);
CREATE INDEX jobs_active ON jobs(bot_id, user_id, update_id)
    WHERE state IN ('pending', 'running');
CREATE INDEX jobs_cleanup ON jobs(finished_at) WHERE finished_at IS NOT NULL;

CREATE TABLE turns (
    bot_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    update_id BIGINT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    PRIMARY KEY (bot_id, user_id, update_id),
    FOREIGN KEY (bot_id, user_id) REFERENCES conversations(bot_id, user_id)
);
