"""Mock Supabase client for testing."""

from __future__ import annotations

import time
from typing import Any


class MockSupabaseAuthUser:
    """Mock Supabase auth user."""

    def __init__(self, user_id: str, email: str, **kwargs: Any):
        self.id = user_id
        self.email = email
        self.user_metadata = kwargs.get("user_metadata", {})
        self.app_metadata = kwargs.get("app_metadata", {})
        self.created_at = kwargs.get("created_at", time.time())
        self.confirmed_at = kwargs.get("confirmed_at", time.time())
        self.email_confirmed_at = kwargs.get("email_confirmed_at", time.time())
        self.phone = kwargs.get("phone")
        self.last_sign_in_at = kwargs.get("last_sign_in_at")
        self.role = kwargs.get("role", "authenticated")
        self.updated_at = kwargs.get("updated_at", time.time())


class MockSupabaseSession:
    """Mock Supabase session."""

    def __init__(
        self, user: MockSupabaseAuthUser, access_token: str, refresh_token: str
    ):
        self.user = user
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = int(time.time()) + 3600  # 1 hour
        self.expires_in = 3600
        self.token_type = "bearer"


class MockSupabaseAuth:
    """Mock Supabase auth client."""

    def __init__(self):
        self.users: dict[str, MockSupabaseAuthUser] = {}
        self.sessions: dict[str, MockSupabaseSession] = {}
        self.current_user: MockSupabaseAuthUser | None = None
        self.current_session: MockSupabaseSession | None = None
        self.error_on_operation: dict[str, bool] = {}

        # Create default test user
        default_user = MockSupabaseAuthUser(
            "test-user-001",
            "test@example.com",
            user_metadata={"full_name": "Test User"},
        )
        self.users[default_user.id] = default_user

    async def sign_up(self, email: str, password: str, **kwargs: Any) -> dict[str, Any]:
        """Mock user signup."""
        if self.error_on_operation.get("sign_up", False):
            raise Exception("Simulated signup error")

        # Check if user already exists
        for user in self.users.values():
            if user.email == email:
                raise Exception("User already registered")

        user_id = f"user_{len(self.users)}"
        user = MockSupabaseAuthUser(
            user_id, email, user_metadata=kwargs.get("data", {})
        )
        self.users[user_id] = user

        # Create session
        access_token = f"access_token_{user_id}"
        refresh_token = f"refresh_token_{user_id}"
        session = MockSupabaseSession(user, access_token, refresh_token)
        self.sessions[access_token] = session

        self.current_user = user
        self.current_session = session

        return {"user": user, "session": session}

    async def sign_in_with_password(self, email: str, password: str) -> dict[str, Any]:
        """Mock user signin."""
        if self.error_on_operation.get("sign_in", False):
            raise Exception("Simulated signin error")

        # Find user by email
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break

        if not user:
            raise Exception("Invalid login credentials")

        # Create session
        access_token = f"access_token_{user.id}_{int(time.time())}"
        refresh_token = f"refresh_token_{user.id}_{int(time.time())}"
        session = MockSupabaseSession(user, access_token, refresh_token)
        self.sessions[access_token] = session

        self.current_user = user
        self.current_session = session

        # Update last sign in
        user.last_sign_in_at = time.time()

        return {"user": user, "session": session}

    async def sign_out(self) -> None:
        """Mock user signout."""
        if self.current_session:
            # Remove session
            self.sessions.pop(self.current_session.access_token, None)

        self.current_user = None
        self.current_session = None

    async def get_user(self, token: str | None = None) -> MockSupabaseAuthUser | None:
        """Get current user or user by token."""
        if token:
            session = self.sessions.get(token)
            return session.user if session else None
        return self.current_user

    async def get_session(self) -> MockSupabaseSession | None:
        """Get current session."""
        return self.current_session

    async def refresh_session(self, refresh_token: str) -> dict[str, Any]:
        """Refresh session."""
        # Find session by refresh token
        session = None
        for s in self.sessions.values():
            if s.refresh_token == refresh_token:
                session = s
                break

        if not session:
            raise Exception("Invalid refresh token")

        # Create new tokens
        new_access_token = f"access_token_{session.user.id}_{int(time.time())}"
        new_refresh_token = f"refresh_token_{session.user.id}_{int(time.time())}"

        # Update session
        old_access_token = session.access_token
        session.access_token = new_access_token
        session.refresh_token = new_refresh_token
        session.expires_at = int(time.time()) + 3600

        # Update sessions dict
        self.sessions.pop(old_access_token, None)
        self.sessions[new_access_token] = session

        return {"user": session.user, "session": session}

    async def update_user(
        self, attributes: dict[str, Any], token: str | None = None
    ) -> MockSupabaseAuthUser:
        """Update user attributes."""
        user = await self.get_user(token)
        if not user:
            raise Exception("User not found")

        # Update user attributes
        for key, value in attributes.items():
            if key == "email":
                user.email = value
            elif key == "user_metadata":
                user.user_metadata.update(value)
            elif key == "app_metadata":
                user.app_metadata.update(value)

        user.updated_at = time.time()
        return user

    async def reset_password_for_email(self, email: str) -> None:
        """Mock password reset."""
        # Just verify user exists
        user_exists = any(u.email == email for u in self.users.values())
        if not user_exists:
            raise Exception("User not found")

    def simulate_error(self, operation: str, should_error: bool = True) -> None:
        """Enable/disable error simulation for specific operations."""
        self.error_on_operation[operation] = should_error


class MockSupabaseTable:
    """Mock Supabase table operations."""

    def __init__(self, table_name: str):
        self.table_name = table_name
        self.data: dict[str, dict[str, Any]] = {}
        self.next_id = 1

    def select(self, columns: str = "*") -> MockSupabaseQuery:
        """Start a select query."""
        return MockSupabaseQuery(self, "select", columns)

    def insert(self, data: dict[str, Any] | list[dict[str, Any]]) -> MockSupabaseQuery:
        """Insert data."""
        if isinstance(data, dict):
            data = [data]

        inserted_data = []
        for item in data:
            if "id" not in item:
                item["id"] = self.next_id
                self.next_id += 1

            item["created_at"] = time.time()
            item["updated_at"] = time.time()
            self.data[str(item["id"])] = item
            inserted_data.append(item)

        query = MockSupabaseQuery(self, "insert")
        query.result_data = inserted_data
        return query

    def update(self, data: dict[str, Any]) -> MockSupabaseQuery:
        """Update data."""
        data["updated_at"] = time.time()
        return MockSupabaseQuery(self, "update", data)

    def delete(self) -> MockSupabaseQuery:
        """Delete data."""
        return MockSupabaseQuery(self, "delete")


class MockSupabaseQuery:
    """Mock Supabase query builder."""

    def __init__(self, table: MockSupabaseTable, operation: str, data: Any = None):
        self.table = table
        self.operation = operation
        self.data = data
        self.filters: list[tuple[str, str, Any]] = []
        self.limit_count: int | None = None
        self.order_by_field: str | None = None
        self.order_ascending = True
        self.result_data: list[dict[str, Any]] | None = None

    def eq(self, column: str, value: Any) -> MockSupabaseQuery:
        """Add equals filter."""
        self.filters.append((column, "eq", value))
        return self

    def neq(self, column: str, value: Any) -> MockSupabaseQuery:
        """Add not equals filter."""
        self.filters.append((column, "neq", value))
        return self

    def gt(self, column: str, value: Any) -> MockSupabaseQuery:
        """Add greater than filter."""
        self.filters.append((column, "gt", value))
        return self

    def gte(self, column: str, value: Any) -> MockSupabaseQuery:
        """Add greater than or equal filter."""
        self.filters.append((column, "gte", value))
        return self

    def lt(self, column: str, value: Any) -> MockSupabaseQuery:
        """Add less than filter."""
        self.filters.append((column, "lt", value))
        return self

    def lte(self, column: str, value: Any) -> MockSupabaseQuery:
        """Add less than or equal filter."""
        self.filters.append((column, "lte", value))
        return self

    def like(self, column: str, pattern: str) -> MockSupabaseQuery:
        """Add like filter."""
        self.filters.append((column, "like", pattern))
        return self

    def in_(self, column: str, values: list[Any]) -> MockSupabaseQuery:
        """Add in filter."""
        self.filters.append((column, "in", values))
        return self

    def limit(self, count: int) -> MockSupabaseQuery:
        """Add limit."""
        self.limit_count = count
        return self

    def order(self, column: str, ascending: bool = True) -> MockSupabaseQuery:
        """Add ordering."""
        self.order_by_field = column
        self.order_ascending = ascending
        return self

    def execute(self) -> dict[str, Any]:
        """Execute the query."""
        if self.result_data is not None:
            return {"data": self.result_data, "error": None}

        # Apply filters and return data
        all_data = list(self.table.data.values())

        # Apply filters
        for column, operator, value in self.filters:
            if operator == "eq":
                all_data = [item for item in all_data if item.get(column) == value]
            elif operator == "neq":
                all_data = [item for item in all_data if item.get(column) != value]
            elif operator == "gt":
                all_data = [item for item in all_data if item.get(column, 0) > value]
            elif operator == "gte":
                all_data = [item for item in all_data if item.get(column, 0) >= value]
            elif operator == "lt":
                all_data = [item for item in all_data if item.get(column, 0) < value]
            elif operator == "lte":
                all_data = [item for item in all_data if item.get(column, 0) <= value]
            elif operator == "like":
                all_data = [
                    item for item in all_data if value in str(item.get(column, ""))
                ]
            elif operator == "in":
                all_data = [item for item in all_data if item.get(column) in value]

        # Apply ordering
        if self.order_by_field:
            all_data.sort(
                key=lambda x: x.get(self.order_by_field, 0),
                reverse=not self.order_ascending,
            )

        # Apply limit
        if self.limit_count:
            all_data = all_data[: self.limit_count]

        # Handle different operations
        if self.operation == "update":
            for item in all_data:
                item.update(self.data)
                self.table.data[str(item["id"])] = item
        elif self.operation == "delete":
            for item in all_data:
                self.table.data.pop(str(item["id"]), None)

        return {"data": all_data, "error": None}


class MockSupabaseClient:
    """Mock Supabase client for testing."""

    def __init__(self, url: str, key: str):
        self.url = url
        self.key = key
        self.auth = MockSupabaseAuth()
        self.tables: dict[str, MockSupabaseTable] = {}

    def table(self, table_name: str) -> MockSupabaseTable:
        """Get or create a table."""
        if table_name not in self.tables:
            self.tables[table_name] = MockSupabaseTable(table_name)
        return self.tables[table_name]

    def clear_all_data(self) -> None:
        """Clear all data."""
        self.tables.clear()
        self.auth.users.clear()
        self.auth.sessions.clear()
        self.auth.current_user = None
        self.auth.current_session = None

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        table_stats = {}
        for name, table in self.tables.items():
            table_stats[name] = len(table.data)

        return {
            "tables": table_stats,
            "users": len(self.auth.users),
            "active_sessions": len(self.auth.sessions),
        }
