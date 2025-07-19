#!/usr/bin/env python3
"""
Comprehensive load testing and performance analysis for Customer Service Bot
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import psutil
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class TestScenario:
    """Load test scenario configuration"""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int = 30
    request_rate_per_second: Optional[int] = None
    endpoints: List[str] = field(default_factory=list)
    payload_generator: Optional[str] = None

@dataclass
class RequestResult:
    """Result of a single request"""
    timestamp: datetime
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    response_size_bytes: int
    error: Optional[str] = None

@dataclass
class LoadTestResults:
    """Comprehensive load test results"""
    scenario_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    requests_per_second: float
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    max_response_time_ms: float
    min_response_time_ms: float
    error_rate_percent: float
    throughput_mb_per_second: float
    system_metrics: Dict[str, Any] = field(default_factory=dict)

class PerformanceTester:
    """
    Comprehensive performance testing suite for Customer Service Bot
    """
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.results: List[RequestResult] = []
        self.system_metrics: List[Dict[str, Any]] = []
        
        # Test data generators
        self.test_messages = [
            "Hello, I need help with my account",
            "What are your business hours?",
            "I'm having trouble with my order",
            "Can you help me reset my password?",
            "I want to cancel my subscription",
            "How do I update my billing information?",
            "What is your return policy?",
            "I'm experiencing technical issues",
            "Can you tell me about your services?",
            "I need to speak with a manager"
        ]
        
        self.test_clients = [
            "test-client-1", "test-client-2", "test-client-3",
            "demo-client", "load-test-client"
        ]
    
    async def create_session(self) -> aiohttp.ClientSession:
        """Create HTTP session with proper headers"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "LoadTester/1.0"
        }
        
        connector = aiohttp.TCPConnector(
            limit=1000,  # Total connection pool size
            limit_per_host=100,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
        
        return aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=timeout
        )
    
    def generate_chat_message_payload(self) -> Dict[str, Any]:
        """Generate realistic chat message payload"""
        import random
        
        return {
            "message": random.choice(self.test_messages),
            "client_id": random.choice(self.test_clients),
            "session_id": f"session_{random.randint(1000, 9999)}",
            "user_id": f"user_{random.randint(100, 999)}",
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "load_test",
                "test_run": True
            }
        }
    
    def generate_admin_request_payload(self) -> Dict[str, Any]:
        """Generate admin API request payload"""
        import random
        
        return {
            "skip": random.randint(0, 100),
            "limit": random.randint(10, 50),
            "status": random.choice(["active", "trial", "suspended"]),
            "tier": random.choice(["free", "starter", "professional"])
        }
    
    async def make_request(
        self,
        session: aiohttp.ClientSession,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> RequestResult:
        """Make a single HTTP request and record metrics"""
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)
        
        try:
            if method.upper() == "GET":
                async with session.get(url, params=payload) as response:
                    content = await response.read()
                    status_code = response.status
                    response_size = len(content)
            elif method.upper() == "POST":
                async with session.post(url, json=payload) as response:
                    content = await response.read()
                    status_code = response.status
                    response_size = len(content)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time_ms = (time.time() - start_time) * 1000
            
            return RequestResult(
                timestamp=timestamp,
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time_ms=response_time_ms,
                response_size_bytes=response_size,
                error=None if 200 <= status_code < 400 else f"HTTP {status_code}"
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            
            return RequestResult(
                timestamp=timestamp,
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time_ms=response_time_ms,
                response_size_bytes=0,
                error=str(e)
            )
    
    async def run_user_simulation(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        scenario: TestScenario,
        results_queue: asyncio.Queue
    ):
        """Simulate a single user's behavior"""
        
        end_time = time.time() + scenario.duration_seconds
        request_interval = 1.0 / (scenario.request_rate_per_second or 1) if scenario.request_rate_per_second else 1.0
        
        while time.time() < end_time:
            # Select random endpoint or use chat endpoint
            if scenario.endpoints:
                endpoint = scenario.endpoints[user_id % len(scenario.endpoints)]
                if "/chat/" in endpoint:
                    payload = self.generate_chat_message_payload()
                    method = "POST"
                elif "/admin/" in endpoint:
                    payload = self.generate_admin_request_payload()
                    method = "GET"
                else:
                    payload = None
                    method = "GET"
            else:
                # Default to chat endpoint
                endpoint = "/api/v1/chat/v2/message"
                payload = self.generate_chat_message_payload()
                method = "POST"
            
            # Make request
            result = await self.make_request(session, method, endpoint, payload)
            await results_queue.put(result)
            
            # Wait before next request
            await asyncio.sleep(request_interval)
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free_gb": disk.free / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv
            }
        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def monitor_system_metrics(self, duration_seconds: int, metrics_queue: asyncio.Queue):
        """Monitor system metrics during load test"""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            metrics = self.collect_system_metrics()
            await metrics_queue.put(metrics)
            await asyncio.sleep(5)  # Collect metrics every 5 seconds
    
    async def run_load_test(self, scenario: TestScenario) -> LoadTestResults:
        """Execute a complete load test scenario"""
        
        print(f"Starting load test: {scenario.name}")
        print(f"Concurrent users: {scenario.concurrent_users}")
        print(f"Duration: {scenario.duration_seconds} seconds")
        print(f"Ramp-up: {scenario.ramp_up_seconds} seconds")
        
        start_time = datetime.now(timezone.utc)
        results_queue = asyncio.Queue()
        metrics_queue = asyncio.Queue()
        
        # Create HTTP session
        session = await self.create_session()
        
        try:
            # Start system monitoring
            monitor_task = asyncio.create_task(
                self.monitor_system_metrics(scenario.duration_seconds + scenario.ramp_up_seconds, metrics_queue)
            )
            
            # Create user simulation tasks
            user_tasks = []
            ramp_up_delay = scenario.ramp_up_seconds / scenario.concurrent_users if scenario.concurrent_users > 0 else 0
            
            for user_id in range(scenario.concurrent_users):
                # Stagger user start times during ramp-up
                start_delay = user_id * ramp_up_delay
                
                task = asyncio.create_task(
                    self._delayed_user_simulation(session, user_id, scenario, results_queue, start_delay)
                )
                user_tasks.append(task)
            
            # Wait for all user simulations to complete
            await asyncio.gather(*user_tasks)
            
            # Stop monitoring
            monitor_task.cancel()
            
            # Collect all results
            request_results = []
            while not results_queue.empty():
                request_results.append(await results_queue.get())
            
            system_metrics = []
            while not metrics_queue.empty():
                system_metrics.append(await metrics_queue.get())
            
        finally:
            await session.close()
        
        end_time = datetime.now(timezone.utc)
        
        # Analyze results
        return self._analyze_results(scenario, start_time, end_time, request_results, system_metrics)
    
    async def _delayed_user_simulation(
        self,
        session: aiohttp.ClientSession,
        user_id: int,
        scenario: TestScenario,
        results_queue: asyncio.Queue,
        delay: float
    ):
        """Start user simulation after delay"""
        await asyncio.sleep(delay)
        await self.run_user_simulation(session, user_id, scenario, results_queue)
    
    def _analyze_results(
        self,
        scenario: TestScenario,
        start_time: datetime,
        end_time: datetime,
        request_results: List[RequestResult],
        system_metrics: List[Dict[str, Any]]
    ) -> LoadTestResults:
        """Analyze load test results and calculate statistics"""
        
        if not request_results:
            return LoadTestResults(
                scenario_name=scenario.name,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                requests_per_second=0.0,
                avg_response_time_ms=0.0,
                p50_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                max_response_time_ms=0.0,
                min_response_time_ms=0.0,
                error_rate_percent=0.0,
                throughput_mb_per_second=0.0,
                system_metrics={}
            )
        
        # Basic statistics
        total_requests = len(request_results)
        successful_requests = len([r for r in request_results if r.error is None])
        failed_requests = total_requests - successful_requests
        
        # Response time statistics
        response_times = [r.response_time_ms for r in request_results]
        response_sizes = [r.response_size_bytes for r in request_results]
        
        duration_seconds = (end_time - start_time).total_seconds()
        requests_per_second = total_requests / duration_seconds if duration_seconds > 0 else 0
        
        # Calculate percentiles
        response_times_sorted = sorted(response_times)
        p50 = response_times_sorted[int(len(response_times_sorted) * 0.5)] if response_times_sorted else 0
        p95 = response_times_sorted[int(len(response_times_sorted) * 0.95)] if response_times_sorted else 0
        p99 = response_times_sorted[int(len(response_times_sorted) * 0.99)] if response_times_sorted else 0
        
        # Throughput calculation
        total_bytes = sum(response_sizes)
        throughput_mb_per_second = (total_bytes / (1024 * 1024)) / duration_seconds if duration_seconds > 0 else 0
        
        # System metrics summary
        system_metrics_summary = {}
        if system_metrics:
            cpu_values = [m.get("cpu_percent", 0) for m in system_metrics if "cpu_percent" in m]
            memory_values = [m.get("memory_percent", 0) for m in system_metrics if "memory_percent" in m]
            
            if cpu_values:
                system_metrics_summary["avg_cpu_percent"] = statistics.mean(cpu_values)
                system_metrics_summary["max_cpu_percent"] = max(cpu_values)
            
            if memory_values:
                system_metrics_summary["avg_memory_percent"] = statistics.mean(memory_values)
                system_metrics_summary["max_memory_percent"] = max(memory_values)
        
        return LoadTestResults(
            scenario_name=scenario.name,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            requests_per_second=requests_per_second,
            avg_response_time_ms=statistics.mean(response_times) if response_times else 0,
            p50_response_time_ms=p50,
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            max_response_time_ms=max(response_times) if response_times else 0,
            min_response_time_ms=min(response_times) if response_times else 0,
            error_rate_percent=(failed_requests / total_requests) * 100 if total_requests > 0 else 0,
            throughput_mb_per_second=throughput_mb_per_second,
            system_metrics=system_metrics_summary
        )
    
    def generate_report(self, results: List[LoadTestResults]) -> str:
        """Generate comprehensive performance test report"""
        
        report = []
        report.append("# Customer Service Bot - Load Test Report")
        report.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append("")
        
        # Summary table
        report.append("## Test Summary")
        report.append("")
        report.append("| Scenario | Users | Requests | RPS | Avg RT (ms) | P95 RT (ms) | Error Rate |")
        report.append("|----------|--------|----------|-----|-------------|-------------|------------|")
        
        for result in results:
            report.append(
                f"| {result.scenario_name} | "
                f"{result.total_requests // (result.end_time - result.start_time).seconds if result.total_requests > 0 else 0} | "
                f"{result.total_requests} | "
                f"{result.requests_per_second:.1f} | "
                f"{result.avg_response_time_ms:.0f} | "
                f"{result.p95_response_time_ms:.0f} | "
                f"{result.error_rate_percent:.1f}% |"
            )
        
        report.append("")
        
        # Detailed results for each scenario
        for result in results:
            report.append(f"## {result.scenario_name}")
            report.append("")
            report.append(f"**Duration:** {(result.end_time - result.start_time).total_seconds():.1f} seconds")
            report.append(f"**Total Requests:** {result.total_requests:,}")
            report.append(f"**Successful Requests:** {result.successful_requests:,}")
            report.append(f"**Failed Requests:** {result.failed_requests:,}")
            report.append(f"**Requests per Second:** {result.requests_per_second:.2f}")
            report.append(f"**Error Rate:** {result.error_rate_percent:.2f}%")
            report.append(f"**Throughput:** {result.throughput_mb_per_second:.2f} MB/s")
            report.append("")
            
            report.append("### Response Time Statistics")
            report.append(f"- **Average:** {result.avg_response_time_ms:.0f} ms")
            report.append(f"- **50th Percentile:** {result.p50_response_time_ms:.0f} ms")
            report.append(f"- **95th Percentile:** {result.p95_response_time_ms:.0f} ms")
            report.append(f"- **99th Percentile:** {result.p99_response_time_ms:.0f} ms")
            report.append(f"- **Minimum:** {result.min_response_time_ms:.0f} ms")
            report.append(f"- **Maximum:** {result.max_response_time_ms:.0f} ms")
            report.append("")
            
            if result.system_metrics:
                report.append("### System Resource Usage")
                for metric, value in result.system_metrics.items():
                    report.append(f"- **{metric.replace('_', ' ').title()}:** {value:.1f}")
                report.append("")
        
        # Performance recommendations
        report.append("## Performance Recommendations")
        report.append("")
        
        for result in results:
            if result.error_rate_percent > 5:
                report.append(f"⚠️ **{result.scenario_name}:** High error rate ({result.error_rate_percent:.1f}%) - investigate application errors")
            
            if result.p95_response_time_ms > 2000:
                report.append(f"⚠️ **{result.scenario_name}:** Slow response times (P95: {result.p95_response_time_ms:.0f}ms) - consider optimization")
            
            if result.system_metrics.get("max_cpu_percent", 0) > 80:
                report.append(f"⚠️ **{result.scenario_name}:** High CPU usage ({result.system_metrics['max_cpu_percent']:.1f}%) - consider scaling")
            
            if result.system_metrics.get("max_memory_percent", 0) > 80:
                report.append(f"⚠️ **{result.scenario_name}:** High memory usage ({result.system_metrics['max_memory_percent']:.1f}%) - check for memory leaks")
        
        return "\n".join(report)
    
    def save_results_to_json(self, results: List[LoadTestResults], filename: str):
        """Save results to JSON file for further analysis"""
        
        data = []
        for result in results:
            data.append({
                "scenario_name": result.scenario_name,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "total_requests": result.total_requests,
                "successful_requests": result.successful_requests,
                "failed_requests": result.failed_requests,
                "requests_per_second": result.requests_per_second,
                "avg_response_time_ms": result.avg_response_time_ms,
                "p50_response_time_ms": result.p50_response_time_ms,
                "p95_response_time_ms": result.p95_response_time_ms,
                "p99_response_time_ms": result.p99_response_time_ms,
                "max_response_time_ms": result.max_response_time_ms,
                "min_response_time_ms": result.min_response_time_ms,
                "error_rate_percent": result.error_rate_percent,
                "throughput_mb_per_second": result.throughput_mb_per_second,
                "system_metrics": result.system_metrics
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

def create_default_scenarios() -> List[TestScenario]:
    """Create default load test scenarios"""
    
    return [
        TestScenario(
            name="Baseline Chat Test",
            concurrent_users=10,
            duration_seconds=60,
            ramp_up_seconds=10,
            request_rate_per_second=2,
            endpoints=["/api/v1/chat/v2/message"]
        ),
        TestScenario(
            name="Moderate Load",
            concurrent_users=50,
            duration_seconds=120,
            ramp_up_seconds=30,
            request_rate_per_second=5,
            endpoints=["/api/v1/chat/v2/message", "/api/v1/health"]
        ),
        TestScenario(
            name="High Load Stress Test",
            concurrent_users=100,
            duration_seconds=180,
            ramp_up_seconds=60,
            request_rate_per_second=10,
            endpoints=[
                "/api/v1/chat/v2/message",
                "/api/v1/admin/clients",
                "/api/v1/health",
                "/api/v1/admin/stats"
            ]
        ),
        TestScenario(
            name="Peak Traffic Simulation",
            concurrent_users=200,
            duration_seconds=300,
            ramp_up_seconds=120,
            request_rate_per_second=15,
            endpoints=["/api/v1/chat/v2/message"]
        )
    ]

async def main():
    """Main entry point for load testing"""
    
    parser = argparse.ArgumentParser(description="Customer Service Bot Load Testing")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--scenarios", nargs="+", help="Specific scenarios to run")
    parser.add_argument("--output-dir", default="load_test_results", help="Output directory for results")
    parser.add_argument("--concurrent-users", type=int, help="Override concurrent users for all scenarios")
    parser.add_argument("--duration", type=int, help="Override duration for all scenarios")
    
    args = parser.parse_args()
    
    # Create output directory
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    tester = PerformanceTester(args.base_url, args.api_key)
    
    # Get scenarios
    scenarios = create_default_scenarios()
    
    # Filter scenarios if specified
    if args.scenarios:
        scenarios = [s for s in scenarios if s.name in args.scenarios]
    
    # Override parameters if specified
    if args.concurrent_users:
        for scenario in scenarios:
            scenario.concurrent_users = args.concurrent_users
    
    if args.duration:
        for scenario in scenarios:
            scenario.duration_seconds = args.duration
    
    # Run tests
    results = []
    for scenario in scenarios:
        print(f"\n{'='*60}")
        result = await tester.run_load_test(scenario)
        results.append(result)
        
        print(f"Completed: {result.scenario_name}")
        print(f"Requests: {result.total_requests}, RPS: {result.requests_per_second:.1f}")
        print(f"Avg Response Time: {result.avg_response_time_ms:.0f}ms")
        print(f"Error Rate: {result.error_rate_percent:.1f}%")
    
    # Generate and save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = tester.generate_report(results)
    report_file = os.path.join(args.output_dir, f"load_test_report_{timestamp}.md")
    with open(report_file, 'w') as f:
        f.write(report)
    
    # Save JSON results
    json_file = os.path.join(args.output_dir, f"load_test_results_{timestamp}.json")
    tester.save_results_to_json(results, json_file)
    
    print(f"\n{'='*60}")
    print("Load testing completed!")
    print(f"Report saved to: {report_file}")
    print(f"Results saved to: {json_file}")
    print(f"\n{report}")

if __name__ == "__main__":
    asyncio.run(main())