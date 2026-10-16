"""One image detector, and it gates a protocol header. Author: Zeno Ren.

The 2026-10-16 GHCP review observed a request logging `image_count=16` and
`has_image=False` at the same time and asked for the two to be reconciled. The
reconciliation matters more than a log inconsistency: `_has_image` also decides
whether `Copilot-Vision-Request: true` is sent, so under-detection means an
image-bearing request goes upstream without the header the upstream expects.

This is **not** a proven root cause for that incident: the small request that
tripped the breaker failed the same way with no images at all. It is a real
defect on its own terms — two detectors over one body disagreeing, with the
narrower one gating a protocol header.

The repository had three detectors with three rule sets:

  * `_has_image`                       top-level messages/input, content must be
                                       a list, types image_url/input_image
  * `estimate_payload`                 recursive, image/image_url/input_image,
                                       plus bare data:image/ strings
  * the context-trim `collect`         recursive, Anthropic image+source, …

Only the first two answer "does this body carry an image", so those two are
unified here. The trim one locates (parent, index) pairs for removal and is a
different question, left alone.
"""
import unittest

import main
import model_capabilities


def vision(body):
    return main.CopilotProxy._has_image(body)


class SharedDetectorTests(unittest.TestCase):
    def test_the_plain_responses_shape_is_still_detected(self):
        body = {'input': [{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': 'https://fixture.invalid/a.png'}]}]}
        self.assertTrue(vision(body))

    def test_the_plain_chat_shape_is_still_detected(self):
        body = {'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'https://fixture.invalid/a.png'}}]}]}
        self.assertTrue(vision(body))

    def test_content_as_a_mapping_is_no_longer_missed(self):
        # `isinstance(content, list)` was the gate; a single content object is
        # legal shape and carried an image straight past it.
        body = {'input': [{'role': 'user', 'content':
                           {'type': 'input_image', 'image_url': 'data:image/png;base64,AAAA'}}]}
        self.assertTrue(vision(body), 'a non-list content block still carries an image')

    def test_an_image_nested_below_the_top_level_is_no_longer_missed(self):
        body = {'input': [{'type': 'function_call_output', 'output': {'content': [
            {'type': 'input_image', 'image_url': 'https://fixture.invalid/tool.png'}]}}]}
        self.assertTrue(vision(body), 'tool output is still part of the request')

    def test_the_anthropic_image_type_is_no_longer_missed(self):
        body = {'messages': [{'role': 'user', 'content': [
            {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                         'data': 'AAAA'}}]}]}
        self.assertTrue(vision(body))

    def test_a_bare_data_url_is_no_longer_missed(self):
        body = {'messages': [{'role': 'user', 'content': ['data:image/png;base64,AAAA']}]}
        self.assertTrue(vision(body))

    def test_a_text_only_body_stays_false(self):
        self.assertFalse(vision({'messages': [{'role': 'user', 'content': 'say ok'}]}))
        self.assertFalse(vision({'input': 'say ok'}))
        self.assertFalse(vision({}))
        self.assertFalse(vision(None) if isinstance(None, dict) else vision({}))

    def test_a_tool_definition_mentioning_an_image_is_not_an_image(self):
        # The false-positive guard that already exists in the estimator: schema
        # subtrees are definitions, not content. Sending the vision header for a
        # tool schema would be the mirror-image defect.
        body = {'messages': [{'role': 'user', 'content': 'hi'}],
                'tools': [{'type': 'function', 'function': {'name': 'f', 'parameters': {
                    'example': 'data:image/png;base64,AAAA',
                    'properties': {'img': {'type': 'input_image'}}}}}]}
        self.assertFalse(vision(body), 'a schema that describes an image carries none')

    def test_the_two_detectors_cannot_disagree_any_more(self):
        """The actual invariant: one body, one answer."""
        bodies = [
            {'messages': [{'role': 'user', 'content': 'say ok'}]},
            {'input': [{'role': 'user', 'content': [{'type': 'input_image',
                                                     'image_url': 'x'}]}]},
            {'input': [{'role': 'user', 'content': {'type': 'input_image',
                                                    'image_url': 'x'}}]},
            {'messages': [{'role': 'user', 'content': ['data:image/png;base64,AAAA']}]},
            {'input': [{'type': 'function_call_output',
                        'output': {'content': [{'type': 'image', 'source': {}}]}}]},
            {'messages': [{'role': 'user', 'content': 'hi'}],
             'tools': [{'parameters': {'example': 'data:image/png;base64,AAAA'}}]},
        ]
        for body in bodies:
            with self.subTest(body=body):
                counted = model_capabilities.estimate_payload(body)['image_count'] > 0
                self.assertEqual(vision(body), counted,
                                 'has_image and image_count must be the same question')

    def test_the_shared_helper_is_what_the_header_gate_uses(self):
        body = {'input': [{'role': 'user', 'content': {'type': 'input_image',
                                                       'image_url': 'x'}}]}
        self.assertTrue(model_capabilities.payload_has_image(body))
        self.assertEqual(vision(body), model_capabilities.payload_has_image(body))

    def test_a_non_mapping_body_is_handled_without_raising(self):
        for value in (None, [], 'text', 7):
            with self.subTest(value=value):
                self.assertFalse(model_capabilities.payload_has_image(value))


class VisionHeaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_header_follows_the_shared_detector(self):
        proxy = main.CopilotProxy.__new__(main.CopilotProxy)
        ep = main.CopilotEndpoint('acct', 'synthetic')

        async def token(_):
            return 'synthetic-session-token'
        proxy.get_session_token = token
        nested = {'input': [{'type': 'function_call_output', 'output': {'content': [
            {'type': 'input_image', 'image_url': 'https://fixture.invalid/t.png'}]}}]}
        headers = await proxy._build_headers(ep, main.CopilotProxy._has_image(nested))
        self.assertEqual(headers.get('Copilot-Vision-Request'), 'true',
                         'an image below the top level still needs the vision header')
        plain = await proxy._build_headers(
            ep, main.CopilotProxy._has_image({'input': 'say ok'}))
        self.assertNotIn('Copilot-Vision-Request', plain)


if __name__ == '__main__':
    unittest.main()
