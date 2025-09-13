-- RLS Setup Verification Script
-- Run this script to verify that RLS is properly configured

-- =============================================================================
-- 1. CHECK RLS FUNCTIONS EXIST
-- =============================================================================

SELECT
    'RLS Functions Check' as check_type,
    proname as function_name,
    CASE
        WHEN proname IS NOT NULL THEN 'EXISTS'
        ELSE 'MISSING'
    END as status
FROM pg_proc
WHERE proname IN ('get_user_role', 'has_role', 'has_any_role')
ORDER BY proname;

-- =============================================================================
-- 2. CHECK RLS IS ENABLED ON TABLES
-- =============================================================================

SELECT
    'RLS Enabled Check' as check_type,
    tablename,
    CASE
        WHEN rowsecurity = true THEN 'ENABLED'
        ELSE 'DISABLED'
    END as rls_status
FROM pg_tables
WHERE schemaname = 'public'
AND tablename IN (
    'system_config', 'audit_log', 'surveys', 'observations',
    'models', 'model_runs', 'detections', 'candidates',
    'difference_runs', 'preprocess_runs', 'processing_jobs',
    'alerts', 'validation_events'
)
ORDER BY tablename;

-- =============================================================================
-- 3. CHECK RLS POLICIES EXIST
-- =============================================================================

SELECT
    'RLS Policies Check' as check_type,
    tablename,
    policyname,
    cmd as operation,
    CASE
        WHEN policyname IS NOT NULL THEN 'EXISTS'
        ELSE 'MISSING'
    END as policy_status
FROM pg_policies
WHERE schemaname = 'public'
ORDER BY tablename, policyname;

-- =============================================================================
-- 4. CHECK AUTHENTICATED ROLE PERMISSIONS
-- =============================================================================

SELECT
    'Permissions Check' as check_type,
    table_name,
    privilege_type,
    CASE
        WHEN privilege_type IS NOT NULL THEN 'GRANTED'
        ELSE 'MISSING'
    END as permission_status
FROM information_schema.table_privileges
WHERE grantee = 'authenticated'
AND table_schema = 'public'
AND table_name IN (
    'system_config', 'audit_log', 'surveys', 'observations',
    'models', 'model_runs', 'detections', 'candidates',
    'difference_runs', 'preprocess_runs', 'processing_jobs',
    'alerts', 'validation_events'
)
ORDER BY table_name, privilege_type;

-- =============================================================================
-- 5. SUMMARY REPORT
-- =============================================================================

SELECT
    'SUMMARY' as report_type,
    'Total Tables with RLS' as metric,
    COUNT(*) as count
FROM pg_tables
WHERE schemaname = 'public'
AND rowsecurity = true
AND tablename IN (
    'system_config', 'audit_log', 'surveys', 'observations',
    'models', 'model_runs', 'detections', 'candidates',
    'difference_runs', 'preprocess_runs', 'processing_jobs',
    'alerts', 'validation_events'
)

UNION ALL

SELECT
    'SUMMARY' as report_type,
    'Total RLS Policies' as metric,
    COUNT(*) as count
FROM pg_policies
WHERE schemaname = 'public'

UNION ALL

SELECT
    'SUMMARY' as report_type,
    'RLS Functions' as metric,
    COUNT(*) as count
FROM pg_proc
WHERE proname IN ('get_user_role', 'has_role', 'has_any_role');
