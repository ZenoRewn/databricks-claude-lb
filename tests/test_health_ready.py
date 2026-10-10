"""Configured-provider readiness regressions; no upstream calls or lifespan."""
import unittest
from unittest.mock import patch

import httpx

import main


class HealthReadyTests(unittest.IsolatedAsyncioTestCase):
    def make_proxy(self, proxy_class, endpoints):
        proxy = object.__new__(proxy_class)
        proxy.load_balancer = main.LoadBalancer(endpoints)
        return proxy

    def setUp(self):
        self.databricks = self.make_proxy(main.ClaudeProxy, [])
        self.azure = self.make_proxy(main.AzureOpenAIProxy, [])
        self.copilot_endpoint = main.CopilotEndpoint(
            "copilot-test", "synthetic", session_token="synthetic-session",
            session_token_expires_at=200,
        )
        self.copilot = self.make_proxy(main.CopilotProxy, [self.copilot_endpoint])

    async def ready(self):
        with patch.multiple(main, proxy=self.databricks, azure_proxy=self.azure,
                            copilot_proxy=self.copilot), \
                patch.object(main.time, "time", return_value=100), \
                patch.object(main.time, "monotonic", return_value=100):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app),
                base_url="http://fixture.invalid",
            ) as client:
                return await client.get("/health/ready")

    def assert_not_ready(self, response, issues):
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {
            "detail": {"status": "not_ready", "issues": issues},
        })

    async def test_copilot_only_ignores_empty_claude_proxy(self):
        for azure in (self.azure, None):
            with self.subTest(empty_azure_proxy=azure is not None):
                self.azure = azure
                response = await self.ready()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": "ready"})

    async def test_configured_databricks_failure_blocks_healthy_copilot(self):
        endpoint = main.WorkspaceEndpoint(
            "db-test", "https://fixture.invalid", "synthetic",
            circuit_open=True, circuit_retry_at=101,
        )
        self.databricks.load_balancer.endpoints = [endpoint]
        self.assert_not_ready(await self.ready(), [
            "databricks: no available endpoints (all circuits open)",
        ])

    async def test_configured_azure_failure_blocks_healthy_copilot(self):
        endpoint = main.AzureOpenAIEndpoint(
            "azure-test", "https://fixture.invalid", "synthetic",
            circuit_open=True, circuit_retry_at=101,
        )
        self.azure.load_balancer.endpoints = [endpoint]
        self.assert_not_ready(await self.ready(), [
            "azure_openai: no available endpoints (all circuits open)",
        ])

    async def test_invalid_or_unhealthy_copilot_remains_not_ready(self):
        cases = (
            {"session_token": None},
            {"session_token_expires_at": 100},
            {"auth_unhealthy": True},
            {"circuit_open": True, "circuit_retry_at": 101},
        )
        for values in cases:
            with self.subTest(values=values):
                endpoint = main.CopilotEndpoint(
                    "copilot-test", "synthetic", session_token="synthetic-session",
                    session_token_expires_at=200,
                )
                for name, value in values.items():
                    setattr(endpoint, name, value)
                self.copilot.load_balancer.endpoints = [endpoint]
                self.assert_not_ready(await self.ready(), [
                    "github_copilot: no healthy endpoint (token invalid or all circuits open)",
                ])

    async def test_healthy_databricks_or_azure_alone_is_ready(self):
        databricks = self.make_proxy(main.ClaudeProxy, [
            main.WorkspaceEndpoint("db-test", "https://fixture.invalid", "synthetic"),
        ])
        azure = self.make_proxy(main.AzureOpenAIProxy, [
            main.AzureOpenAIEndpoint("azure-test", "https://fixture.invalid", "synthetic"),
        ])
        empty_databricks, empty_azure = self.databricks, self.azure
        self.copilot = None
        for db, az in ((databricks, empty_azure), (empty_databricks, azure)):
            with self.subTest(provider="databricks" if db is databricks else "azure"):
                self.databricks, self.azure = db, az
                response = await self.ready()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"status": "ready"})

    async def test_one_healthy_endpoint_per_configured_provider_is_ready(self):
        self.databricks.load_balancer.endpoints = [
            main.WorkspaceEndpoint(
                "db-open", "https://fixture.invalid", "synthetic",
                circuit_open=True, circuit_retry_at=101,
            ),
            main.WorkspaceEndpoint("db-closed", "https://fixture.invalid", "synthetic"),
        ]
        self.azure.load_balancer.endpoints = [
            main.AzureOpenAIEndpoint(
                "azure-open", "https://fixture.invalid", "synthetic",
                circuit_open=True, circuit_retry_at=101,
            ),
            main.AzureOpenAIEndpoint("azure-closed", "https://fixture.invalid", "synthetic"),
        ]
        self.copilot.load_balancer.endpoints.insert(0, main.CopilotEndpoint(
            "copilot-invalid", "synthetic", auth_unhealthy=True,
        ))
        response = await self.ready()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})

    async def test_recovery_ready_does_not_claim_or_close_a_trial(self):
        self.databricks.load_balancer.endpoints = [
            main.WorkspaceEndpoint("db-test", "https://fixture.invalid", "synthetic"),
        ]
        self.azure.load_balancer.endpoints = [
            main.AzureOpenAIEndpoint("azure-test", "https://fixture.invalid", "synthetic"),
        ]
        providers = (self.databricks, self.azure, self.copilot)
        for provider in providers:
            endpoint = provider.load_balancer.endpoints[0]
            endpoint.circuit_open = True
            endpoint.circuit_retry_at = 100
            endpoint.half_open_in_flight = True
        response = await self.ready()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})
        for provider in providers:
            endpoint = provider.load_balancer.endpoints[0]
            self.assertTrue(endpoint.circuit_open)
            self.assertTrue(endpoint.half_open_in_flight)
            self.assertEqual(endpoint.circuit_generation, 0)
            self.assertEqual(provider.load_balancer._attempts, {})

    async def test_no_configured_routes_is_not_ready(self):
        empty_copilot = self.make_proxy(main.CopilotProxy, [])
        cases = (
            (None, None, None),
            (self.databricks, None, None),
            (self.databricks, self.azure, empty_copilot),
        )
        for databricks, azure, copilot in cases:
            with self.subTest(providers=(databricks, azure, copilot)):
                self.databricks, self.azure, self.copilot = databricks, azure, copilot
                self.assert_not_ready(await self.ready(), [
                    "routing: no configured endpoints",
                ])
