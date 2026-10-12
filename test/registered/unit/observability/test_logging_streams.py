# SPDX-License-Identifier: Apache-2.0

import io
import logging
import logging.config
import unittest
from contextlib import redirect_stderr, redirect_stdout

from sglang.logging_utils import configure_root_logger, configure_uvicorn_streams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestLoggingStreams(CustomTestCase):
    """Prevent collectors from treating ordinary stderr output as errors."""

    def setUp(self):
        self.root = logging.getLogger()
        self.handlers = self.root.handlers[:]
        self.level = self.root.level
        # Keep the test runner's handlers from being closed by force=True.
        self.root.handlers = []
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(redirect_stderr(self.stderr))
        self.formatter = logging.Formatter("%(levelname)s %(message)s")

    def tearDown(self):
        for handler in self.root.handlers[:]:
            self.root.removeHandler(handler)
            handler.close()
        self.root.handlers = self.handlers
        self.root.setLevel(self.level)
        super().tearDown()

    def test_root_routes_all_levels_once(self):
        configure_root_logger(logging.DEBUG, self.formatter)
        for level in ("debug", "info", "warning", "error", "critical"):
            getattr(self.root, level)(level)
        self.assertEqual(
            self.stdout.getvalue(), "DEBUG debug\nINFO info\nWARNING warning\n"
        )
        self.assertEqual(self.stderr.getvalue(), "ERROR error\nCRITICAL critical\n")

    def test_threshold_and_reconfiguration(self):
        configure_root_logger(logging.DEBUG, self.formatter)
        configure_root_logger(logging.WARNING, self.formatter)
        self.root.info("hidden")
        self.root.warning("visible")
        self.root.error("failure")
        self.assertEqual(self.stdout.getvalue(), "WARNING visible\n")
        self.assertEqual(self.stderr.getvalue(), "ERROR failure\n")

    def test_exception_traceback_only_on_stderr(self):
        configure_root_logger(logging.INFO, self.formatter)
        try:
            raise ValueError("stream regression")
        except ValueError:
            self.root.exception("request failed")
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertIn("ERROR request failed\nTraceback", self.stderr.getvalue())
        self.assertIn("ValueError: stream regression", self.stderr.getvalue())

    def test_early_logging_preserves_application_handlers(self):
        application_output = io.StringIO()
        handler = logging.StreamHandler(application_output)
        self.root.addHandler(handler)
        self.root.setLevel(logging.INFO)
        configure_root_logger(logging.WARNING, self.formatter, force=False)
        self.root.info("application message")
        self.assertEqual(application_output.getvalue(), "application message\n")
        self.assertEqual(self.stdout.getvalue(), "")
        self.assertEqual(self.stderr.getvalue(), "")

    def test_uvicorn_runtime_and_access_streams(self):
        from copy import deepcopy

        from uvicorn.config import LOGGING_CONFIG

        names = ("uvicorn", "uvicorn.error", "uvicorn.access")
        loggers = [logging.getLogger(name) for name in names]
        states = [
            (logger.handlers[:], logger.level, logger.propagate, logger.disabled)
            for logger in loggers
        ]
        config = deepcopy(LOGGING_CONFIG)
        try:
            configure_uvicorn_streams(config)
            configure_uvicorn_streams(config)
            logging.config.dictConfig(config)
            runtime = logging.getLogger("uvicorn.error")
            runtime.info("server ready")
            runtime.warning("server warning")
            runtime.error("server failed")
            logging.getLogger("uvicorn.access").info(
                '%s - "%s %s HTTP/%s" %d',
                "127.0.0.1:1234",
                "GET",
                "/health",
                "1.1",
                200,
            )
            self.assertIn("server ready", self.stdout.getvalue())
            self.assertIn("server warning", self.stdout.getvalue())
            self.assertIn("/health", self.stdout.getvalue())
            self.assertNotIn("server failed", self.stdout.getvalue())
            self.assertEqual(self.stderr.getvalue().count("server failed"), 1)
            self.assertNotIn("server ready", self.stderr.getvalue())
            self.assertNotIn("server warning", self.stderr.getvalue())
            self.assertNotIn("/health", self.stderr.getvalue())
        finally:
            for logger, (handlers, level, propagate, disabled) in zip(loggers, states):
                for handler in logger.handlers:
                    handler.close()
                logger.handlers = handlers
                logger.setLevel(level)
                logger.propagate = propagate
                logger.disabled = disabled


if __name__ == "__main__":
    unittest.main()
