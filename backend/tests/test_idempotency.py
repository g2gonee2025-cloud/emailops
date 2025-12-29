import unittest
import uuid

import pytest
from cortex.ingestion.mailroom import _generate_stable_id


class TestIdempotency(unittest.TestCase):
    def test_stable_id_generation(self):
        # Verify that generating an ID twice with same inputs produces same UUID
        ns = uuid.NAMESPACE_DNS
        id1 = _generate_stable_id(ns, "test", "foo")
        id2 = _generate_stable_id(ns, "test", "foo")
        id3 = _generate_stable_id(ns, "test", "bar")

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertIsInstance(id1, uuid.UUID)


@pytest.mark.asyncio
class TestIdempotencyAsync:
    """Test idempotency of conversation ingestion using live infrastructure."""

    async def test_stable_ids_for_same_folder(self):
        """Test that same folder name produces same conversation ID across runs."""
        # Import the actual function that generates stable IDs
        from cortex.ingestion.mailroom import _generate_stable_id

        ns = uuid.NAMESPACE_DNS
        tenant = "test-tenant"
        folder = "2024-01-01_important_meeting"

        # Generate IDs multiple times
        id1 = _generate_stable_id(ns, tenant, folder)
        id2 = _generate_stable_id(ns, tenant, folder)

        assert id1 == id2, "Same inputs should produce same ID"
        assert isinstance(id1, uuid.UUID)

    async def test_different_folders_produce_different_ids(self):
        """Test that different folder names produce different conversation IDs."""
        from cortex.ingestion.mailroom import _generate_stable_id

        ns = uuid.NAMESPACE_DNS
        tenant = "test-tenant"

        id1 = _generate_stable_id(ns, tenant, "folder_a")
        id2 = _generate_stable_id(ns, tenant, "folder_b")

        assert id1 != id2, "Different folders should produce different IDs"

    async def test_different_tenants_produce_different_ids(self):
        """Test that different tenants produce different conversation IDs."""
        from cortex.ingestion.mailroom import _generate_stable_id

        ns = uuid.NAMESPACE_DNS
        folder = "same_folder"

        id1 = _generate_stable_id(ns, "tenant_a", folder)
        id2 = _generate_stable_id(ns, "tenant_b", folder)

        assert id1 != id2, "Different tenants should produce different IDs"
