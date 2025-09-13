-- User Role Setup for AstrID
-- Run this script in Supabase SQL Editor to set up user roles
-- This script works with the RLS system to provide role-based access control

-- Function to set user role (if not already exists)
CREATE OR REPLACE FUNCTION set_user_role(user_id UUID, role_name TEXT)
RETURNS VOID AS $$
BEGIN
    -- Update the role column directly in auth.users
    UPDATE auth.users
    SET role = role_name
    WHERE id = user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant permissions
GRANT EXECUTE ON FUNCTION set_user_role(UUID, TEXT) TO authenticated;

-- =============================================================================
-- USAGE INSTRUCTIONS
-- =============================================================================

-- 1. First, find your user ID:
-- SELECT id, email, raw_user_meta_data FROM auth.users;

-- 2. Then set roles (replace with your actual user IDs):
-- SELECT set_user_role('your-user-id-here', 'admin');
-- SELECT set_user_role('another-user-id-here', 'researcher');
-- SELECT set_user_role('third-user-id-here', 'analyst');

-- =============================================================================
-- AVAILABLE ROLES
-- =============================================================================

-- - admin: Full system access (all tables, all operations)
-- - researcher: Full research operations access (models, runs, processing)
-- - analyst: Read/write analysis data access (detections, candidates, alerts)
-- - viewer: Read access to most data (surveys, observations)
-- - guest: Read-only public data (default for new users)

-- =============================================================================
-- ROLE PERMISSIONS SUMMARY
-- =============================================================================

-- ADMIN:
--   - All tables: SELECT, INSERT, UPDATE, DELETE
--   - System configuration access
--   - Audit log access

-- RESEARCHER:
--   - Models, model_runs: SELECT, INSERT, UPDATE, DELETE
--   - Processing tables: SELECT, INSERT, UPDATE, DELETE
--   - Surveys, observations: SELECT, INSERT, UPDATE, DELETE
--   - Detections, candidates: SELECT, INSERT, UPDATE, DELETE
--   - Alerts, validation_events: SELECT, INSERT, UPDATE, DELETE

-- ANALYST:
--   - Detections, candidates: SELECT
--   - Alerts, validation_events: SELECT
--   - Surveys, observations: SELECT

-- VIEWER:
--   - Surveys, observations: SELECT

-- GUEST:
--   - No access (default)
