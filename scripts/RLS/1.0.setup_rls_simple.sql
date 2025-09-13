-- Simple RLS Setup for AstrID
-- This is a safer, step-by-step approach to avoid Supabase warnings

-- =============================================================================
-- STEP 1: Create RLS Helper Functions
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
-- STEP 2: Create User Role Management Function
-- =============================================================================

-- Function to set user role (directly updates the role column)
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
