import unittest
from effort_compat import preserve_native_effort, effort_response_headers
class NativeEffort(unittest.TestCase):
    def test_supported_effort_survives_without_forwarding_unverified_format(self):
        for effort in ('low','medium','high'):
            body={'model':'databricks-claude-opus-5','thinking':{'type':'adaptive'},'output_config':{'effort':effort,'format':{'type':'json_schema'}}}
            preserve_native_effort(body)
            self.assertEqual(body['output_config'],{'effort':effort})
            self.assertEqual(body['thinking'],{'type':'adaptive'})
            self.assertEqual(effort_response_headers(body),{'x-claude-effort-forwarded':effort})
    def test_invalid_values_reach_upstream_and_cannot_become_headers(self):
        for effort in ('__invalid__','low\r\nx: y',{'bad':'type'},None):
            body={'model':'databricks-claude-opus-5','output_config':{'effort':effort}}
            preserve_native_effort(body)
            self.assertEqual(body['output_config']['effort'],effort)
            self.assertEqual(effort_response_headers(body),{})
    def test_other_models_keep_legacy_compatibility(self):
        for model in ('databricks-claude-opus-4-5','databricks-claude-sonnet-4-5','unknown'):
            body={'model':model,'output_config':{'effort':'high','format':{}}}
            preserve_native_effort(body)
            self.assertNotIn('output_config',body)
    def test_missing_or_format_only_config_is_not_invented(self):
        for config in ({'format':{}},None,'bad'):
            body={'model':'databricks-claude-opus-5','output_config':config}
            preserve_native_effort(body)
            self.assertNotIn('output_config',body)
if __name__=='__main__':unittest.main()
