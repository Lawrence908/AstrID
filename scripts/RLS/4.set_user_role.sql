-- First find your user ID
SELECT id, email, role FROM auth.users;

-- Then set your role (this should work now)
SELECT set_user_role('your-user-id-here', 'admin');
