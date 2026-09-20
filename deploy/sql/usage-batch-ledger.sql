-- Author: Zeno Ren
-- Additive schema for atomic, idempotent usage flushes. Review before production.
-- Keep ledger IDs while any writer may retry an old batch. Do not truncate this
-- table as part of ordinary usage_daily retention. Event payloads are cleared
-- at the usage retention cutoff; only batch ID/date/hash receipts remain.
CREATE TABLE IF NOT EXISTS usage_batch_ledger (
    batch_id CHAR(36) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    event_date DATE NOT NULL,
    payload_sha256 CHAR(64) CHARACTER SET ascii COLLATE ascii_bin NOT NULL,
    payload JSON NULL,
    created_at TIMESTAMP(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    PRIMARY KEY (batch_id),
    KEY usage_batch_event_date (event_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
