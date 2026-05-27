import unittest
import os
import importlib
from unittest.mock import patch
import config

class TestConfig(unittest.TestCase):
    def test_default_config_values(self):
        """Test DefaultConfig with default environment variables."""
        # Clear relevant environment variables to test defaults
        with patch.dict(os.environ, {}, clear=True):
            importlib.reload(config)
            self.assertEqual(config.DefaultConfig.PORT, 3978)
            self.assertEqual(config.DefaultConfig.APP_ID, "")
            self.assertIsNone(config.DefaultConfig.APP_PASSWORD)
            self.assertEqual(config.DefaultConfig.APP_TYPE, "MultiTenant")
            self.assertEqual(config.DefaultConfig.APP_TENANTID, "")

    def test_custom_config_values(self):
        """Test DefaultConfig with custom environment variables."""
        custom_env = {
            "MicrosoftAppId": "my_id",
            "MicrosoftAppPassword": "my_password",
            "MicrosoftAppType": "SingleTenant",
            "MicrosoftAppTenantId": "my_tenant_id"
        }
        with patch.dict(os.environ, custom_env, clear=True):
            importlib.reload(config)
            self.assertEqual(config.DefaultConfig.APP_ID, "my_id")
            self.assertEqual(config.DefaultConfig.APP_PASSWORD, "my_password")
            self.assertEqual(config.DefaultConfig.APP_TYPE, "SingleTenant")
            self.assertEqual(config.DefaultConfig.APP_TENANTID, "my_tenant_id")

if __name__ == "__main__":
    unittest.main()
