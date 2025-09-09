-- This should now work without errors
SELECT get_user_role();
SELECT has_role('admin');
SELECT has_any_role('admin', 'researcher');
