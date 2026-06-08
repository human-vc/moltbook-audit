"""Property-based tests for rate limiting.

Feature: molt-dynamics-analysis
Property 4: Rate Limiting Enforcement
Validates: Requirements 1.2
"""

import sys
import time
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from molt_dynamics.scraper import RateLimiter


class TestRateLimitingEnforcement:
    """Property 4: Rate Limiting Enforcement
    
    For any sequence of N consecutive API requests, the elapsed time between
    the first and last request SHALL be at least (N-1) × rate_limit_delay seconds.
    """
    
    @given(
        delay=st.floats(min_value=0.05, max_value=0.2),
        n_requests=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=20, deadline=None)
    def test_minimum_elapsed_time(self, delay: float, n_requests: int):
        """Total elapsed time should be at least (N-1) * delay."""
        limiter = RateLimiter(delay=delay)
        
        start_time = time.time()
        
        for _ in range(n_requests):
            limiter.wait()
        
        elapsed = time.time() - start_time
        expected_minimum = (n_requests - 1) * delay
        
        tolerance = 0.01 * n_requests
        
        assert elapsed >= expected_minimum - tolerance, (
            f"Elapsed {elapsed:.4f}s < expected minimum {expected_minimum:.4f}s "
            f"for {n_requests} requests with {delay}s delay"
        )
    
    @given(delay=st.floats(min_value=0.05, max_value=0.3))
    @settings(max_examples=20, deadline=None)
    def test_consecutive_requests_respect_delay(self, delay: float):
        """Two consecutive requests should be separated by at least delay."""
        limiter = RateLimiter(delay=delay)
        
        limiter.wait()
        time1 = limiter.get_last_request_time()
        
        limiter.wait()
        time2 = limiter.get_last_request_time()
        
        actual_delay = time2 - time1
        
        tolerance = 0.01
        
        assert actual_delay >= delay - tolerance, (
            f"Actual delay {actual_delay:.4f}s < configured delay {delay:.4f}s"
        )
    
    def test_first_request_no_wait(self):
        """First request should not wait."""
        limiter = RateLimiter(delay=1.0)
        
        start = time.time()
        wait_time = limiter.wait()
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"First request waited {elapsed}s"
        assert wait_time == 0.0, f"First request reported wait time {wait_time}"
    
    def test_request_after_delay_no_wait(self):
        """Request after delay period should not wait."""
        delay = 0.1
        limiter = RateLimiter(delay=delay)
        
        limiter.wait()
        
        time.sleep(delay * 1.5)
        
        start = time.time()
        wait_time = limiter.wait()
        elapsed = time.time() - start
        
        assert elapsed < 0.05, f"Request after delay waited {elapsed}s"
        assert wait_time == 0.0, f"Request after delay reported wait time {wait_time}"
    
    @given(delay=st.floats(min_value=0.05, max_value=0.2))
    @settings(max_examples=10, deadline=None)
    def test_wait_returns_actual_wait_time(self, delay: float):
        """wait() should return the actual time waited."""
        limiter = RateLimiter(delay=delay)
        
        wait1 = limiter.wait()
        assert wait1 == 0.0
        
        wait2 = limiter.wait()
        
        tolerance = 0.02
        assert abs(wait2 - delay) < tolerance, (
            f"Reported wait {wait2:.4f}s differs from delay {delay:.4f}s"
        )


class TestRateLimiterThreadSafety:
    """Tests for thread-safe rate limiting."""
    
    def test_concurrent_requests_respect_delay(self):
        """Concurrent requests from multiple threads should still respect delay."""
        import threading
        
        delay = 0.1
        limiter = RateLimiter(delay=delay)
        n_threads = 3
        n_requests_per_thread = 2
        
        request_times = []
        lock = threading.Lock()
        
        def make_requests():
            for _ in range(n_requests_per_thread):
                limiter.wait()
                with lock:
                    request_times.append(time.time())
        
        threads = [threading.Thread(target=make_requests) for _ in range(n_threads)]
        
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        
        total_requests = n_threads * n_requests_per_thread
        expected_minimum = (total_requests - 1) * delay
        
        tolerance = 0.05 * total_requests
        
        assert elapsed >= expected_minimum - tolerance, (
            f"Elapsed {elapsed:.4f}s < expected {expected_minimum:.4f}s "
            f"for {total_requests} concurrent requests"
        )


class TestRateLimiterEdgeCases:
    """Edge case tests for rate limiter."""
    
    def test_zero_delay(self):
        """Zero delay should allow immediate requests."""
        limiter = RateLimiter(delay=0.0)
        
        start = time.time()
        for _ in range(10):
            limiter.wait()
        elapsed = time.time() - start
        
        assert elapsed < 0.1, f"Zero delay took {elapsed}s for 10 requests"
    
    def test_very_small_delay(self):
        """Very small delays should still be enforced."""
        delay = 0.01
        limiter = RateLimiter(delay=delay)
        
        start = time.time()
        for _ in range(5):
            limiter.wait()
        elapsed = time.time() - start
        
        expected = 4 * delay
        assert elapsed >= expected - 0.01, (
            f"Elapsed {elapsed:.4f}s < expected {expected:.4f}s"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
