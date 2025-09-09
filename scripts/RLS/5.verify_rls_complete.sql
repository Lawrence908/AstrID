-- Complete RLS Verification Script
-- Run this to confirm everything is working

-- =============================================================================
-- 1. TEST RLS FUNCTIONS
-- =============================================================================

SELECT
    'RLS Functions Test' as test_type,
    get_user_role() as current_user_role,
    has_role('admin') as has_admin_role,
    has_any_role('admin', 'researcher') as has_admin_or_researcher;

-- =============================================================================
-- 2. CHECK RLS STATUS ON ALL TABLES
-- =============================================================================

SELECT
    'RLS Status Check' as test_type,
    tablename,
    CASE
        WHEN rowsecurity = true THEN '✅ ENABLED'
        ELSE '❌ DISABLED'
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
-- 3. CHECK POLICIES ARE CREATED
-- =============================================================================

SELECT
    'Policy Count Check' as test_type,
    COUNT(*) as total_policies,
    COUNT(DISTINCT tablename) as tables_with_policies
FROM pg_policies
WHERE schemaname = 'public';

-- =============================================================================
-- 4. TEST TABLE ACCESS (Should work for admin)
-- =============================================================================

-- Test access to different table types
SELECT 'Table Access Test' as test_type, 'system_config' as table_name, COUNT(*) as accessible_rows FROM system_config
UNION ALL
SELECT 'Table Access Test' as test_type, 'surveys' as table_name, COUNT(*) as accessible_rows FROM surveys
UNION ALL
SELECT 'Table Access Test' as test_type, 'models' as table_name, COUNT(*) as accessible_rows FROM models;

-- =============================================================================
-- 5. SUMMARY
-- =============================================================================

SELECT
    'SUMMARY' as report_type,
    'RLS Setup Status' as metric,
    CASE
        WHEN (SELECT COUNT(*) FROM pg_proc WHERE proname IN ('get_user_role', 'has_role', 'has_any_role')) = 3
        AND (SELECT COUNT(*) FROM pg_tables WHERE schemaname = 'public' AND rowsecurity = true) >= 10
        AND (SELECT COUNT(*) FROM pg_policies WHERE schemaname = 'public') >= 15
        THEN '✅ COMPLETE - RLS is fully configured'
        ELSE '❌ INCOMPLETE - Some components missing'
    END as status;
