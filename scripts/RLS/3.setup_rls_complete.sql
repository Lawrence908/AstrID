-- Complete RLS Setup for AstrID
-- Run this script in Supabase SQL Editor to set up Row Level Security
-- This replaces the automatic Alembic approach with a manual, reliable setup

-- =============================================================================
-- 1. CREATE HELPER FUNCTIONS
-- =============================================================================

-- Function to get user role from JWT
CREATE OR REPLACE FUNCTION get_user_role()
RETURNS TEXT AS $$
BEGIN
    RETURN COALESCE(
        (auth.jwt() ->> 'role')::text,
        'guest'
    );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if user has specific role
CREATE OR REPLACE FUNCTION has_role(role_name TEXT)
RETURNS BOOLEAN AS $$
BEGIN
    RETURN get_user_role() = role_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if user has any of the specified roles
CREATE OR REPLACE FUNCTION has_any_role(VARIADIC role_names TEXT[])
RETURNS BOOLEAN AS $$
BEGIN
    RETURN get_user_role() = ANY(role_names);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant permissions to authenticated users
GRANT EXECUTE ON FUNCTION get_user_role() TO authenticated;
GRANT EXECUTE ON FUNCTION has_role(TEXT) TO authenticated;
GRANT EXECUTE ON FUNCTION has_any_role(TEXT[]) TO authenticated;

-- =============================================================================
-- 2. ENABLE RLS ON ALL TABLES
-- =============================================================================

-- Core system tables
ALTER TABLE system_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;

-- User data tables
ALTER TABLE surveys ENABLE ROW LEVEL SECURITY;
ALTER TABLE observations ENABLE ROW LEVEL SECURITY;

-- ML/Analysis tables
ALTER TABLE models ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidates ENABLE ROW LEVEL SECURITY;

-- Processing tables
ALTER TABLE difference_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE preprocess_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_jobs ENABLE ROW LEVEL SECURITY;

-- Alert/validation tables
ALTER TABLE alerts ENABLE ROW LEVEL SECURITY;
ALTER TABLE validation_events ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- 3. CREATE RLS POLICIES
-- =============================================================================

-- SYSTEM_CONFIG: Admin only
DROP POLICY IF EXISTS system_config_admin_only ON system_config;
CREATE POLICY system_config_admin_only ON system_config
    FOR ALL TO authenticated
    USING (has_role('admin'))
    WITH CHECK (has_role('admin'));

-- AUDIT_LOG: Admin only
DROP POLICY IF EXISTS audit_log_admin_only ON audit_log;
CREATE POLICY audit_log_admin_only ON audit_log
    FOR ALL TO authenticated
    USING (has_role('admin'))
    WITH CHECK (has_role('admin'));

-- SURVEYS: Viewers can read, researchers can modify
DROP POLICY IF EXISTS surveys_viewers_can_read ON surveys;
CREATE POLICY surveys_viewers_can_read ON surveys
    FOR SELECT TO authenticated
    USING (has_any_role('admin', 'researcher', 'analyst', 'viewer'));

DROP POLICY IF EXISTS surveys_researchers_can_modify ON surveys;
CREATE POLICY surveys_researchers_can_modify ON surveys
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- OBSERVATIONS: Viewers can read, researchers can modify
DROP POLICY IF EXISTS observations_viewers_can_read ON observations;
CREATE POLICY observations_viewers_can_read ON observations
    FOR SELECT TO authenticated
    USING (has_any_role('admin', 'researcher', 'analyst', 'viewer'));

DROP POLICY IF EXISTS observations_researchers_can_modify ON observations;
CREATE POLICY observations_researchers_can_modify ON observations
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- MODELS: Researchers can access
DROP POLICY IF EXISTS models_researchers_can_access ON models;
CREATE POLICY models_researchers_can_access ON models
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- MODEL_RUNS: Researchers can access
DROP POLICY IF EXISTS model_runs_researchers_can_access ON model_runs;
CREATE POLICY model_runs_researchers_can_access ON model_runs
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- DETECTIONS: Analysts can read, researchers can modify
DROP POLICY IF EXISTS detections_analysts_can_read ON detections;
CREATE POLICY detections_analysts_can_read ON detections
    FOR SELECT TO authenticated
    USING (has_any_role('admin', 'researcher', 'analyst'));

DROP POLICY IF EXISTS detections_researchers_can_modify ON detections;
CREATE POLICY detections_researchers_can_modify ON detections
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- CANDIDATES: Analysts can read, researchers can modify
DROP POLICY IF EXISTS candidates_analysts_can_read ON candidates;
CREATE POLICY candidates_analysts_can_read ON candidates
    FOR SELECT TO authenticated
    USING (has_any_role('admin', 'researcher', 'analyst'));

DROP POLICY IF EXISTS candidates_researchers_can_modify ON candidates;
CREATE POLICY candidates_researchers_can_modify ON candidates
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- DIFFERENCE_RUNS: Researchers can access
DROP POLICY IF EXISTS difference_runs_researchers_can_access ON difference_runs;
CREATE POLICY difference_runs_researchers_can_access ON difference_runs
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- PREPROCESS_RUNS: Researchers can access
DROP POLICY IF EXISTS preprocess_runs_researchers_can_access ON preprocess_runs;
CREATE POLICY preprocess_runs_researchers_can_access ON preprocess_runs
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- PROCESSING_JOBS: Researchers can access
DROP POLICY IF EXISTS processing_jobs_researchers_can_access ON processing_jobs;
CREATE POLICY processing_jobs_researchers_can_access ON processing_jobs
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- ALERTS: Analysts can read, researchers can modify
DROP POLICY IF EXISTS alerts_analysts_can_read ON alerts;
CREATE POLICY alerts_analysts_can_read ON alerts
    FOR SELECT TO authenticated
    USING (has_any_role('admin', 'researcher', 'analyst'));

DROP POLICY IF EXISTS alerts_researchers_can_modify ON alerts;
CREATE POLICY alerts_researchers_can_modify ON alerts
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- VALIDATION_EVENTS: Analysts can read, researchers can modify
DROP POLICY IF EXISTS validation_events_analysts_can_read ON validation_events;
CREATE POLICY validation_events_analysts_can_read ON validation_events
    FOR SELECT TO authenticated
    USING (has_any_role('admin', 'researcher', 'analyst'));

DROP POLICY IF EXISTS validation_events_researchers_can_modify ON validation_events;
CREATE POLICY validation_events_researchers_can_modify ON validation_events
    FOR ALL TO authenticated
    USING (has_any_role('admin', 'researcher'))
    WITH CHECK (has_any_role('admin', 'researcher'));

-- =============================================================================
-- 4. GRANT PERMISSIONS TO AUTHENTICATED ROLE
-- =============================================================================

-- Grant basic permissions to authenticated users
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- =============================================================================
-- 5. VERIFICATION QUERIES
-- =============================================================================

-- Check RLS is enabled on all tables
SELECT schemaname, tablename, rowsecurity
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY tablename;

-- Check policies are created
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;
