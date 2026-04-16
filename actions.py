"""Phase 4: Smart plug control and action triggers."""

import asyncio
import time
import logging
import threading
from datetime import datetime

from kasa import Discover

from config import (
    SMART_PLUG_IP,
    LIGHT_ON_DURATION,
    ACTION_COOLDOWN,
    ALERT_START_HOUR,
    ALERT_END_HOUR,
)

logger = logging.getLogger(__name__)


class ActionManager:
    """Controls a smart plug in response to person detections.

    Features:
    - Cooldown period to prevent rapid toggling
    - Time-of-day schedule (only triggers during configured hours)
    - Auto-off timer after configurable duration
    """

    def __init__(
        self,
        plug_ip: str = SMART_PLUG_IP,
        cooldown: float = ACTION_COOLDOWN,
        light_duration: float = LIGHT_ON_DURATION,
    ):
        self.plug_ip = plug_ip
        self.cooldown = cooldown
        self.light_duration = light_duration
        self.last_trigger_time = 0.0
        self._off_timer = None
        self._loop = asyncio.new_event_loop()

    def is_alert_hours(self) -> bool:
        """Check if current time falls within the alert window."""
        hour = datetime.now().hour
        if ALERT_START_HOUR > ALERT_END_HOUR:
            # Window crosses midnight (e.g., 23:00 - 06:00)
            return hour >= ALERT_START_HOUR or hour < ALERT_END_HOUR
        else:
            return ALERT_START_HOUR <= hour < ALERT_END_HOUR

    def on_person_detected(self, num_persons: int, snapshot_path: str = None):
        """Called by the pipeline when a person is detected.

        Respects cooldown and schedule before triggering the light.
        """
        if not self.is_alert_hours():
            return

        now = time.time()
        if now - self.last_trigger_time < self.cooldown:
            return

        self.last_trigger_time = now
        logger.info(
            "ACTION: %d person(s) detected. Turning on light.", num_persons
        )

        self._loop.run_until_complete(self._turn_on_light())
        self._schedule_off()

    async def _turn_on_light(self):
        """Turn on the smart plug."""
        try:
            dev = await Discover.discover_single(self.plug_ip)
            await dev.turn_on()
            await dev.update()
            logger.info("Smart plug ON at %s", self.plug_ip)
        except Exception:
            logger.exception("Failed to turn on smart plug at %s", self.plug_ip)

    async def _turn_off_light(self):
        """Turn off the smart plug."""
        try:
            dev = await Discover.discover_single(self.plug_ip)
            await dev.turn_off()
            await dev.update()
            logger.info("Smart plug OFF at %s", self.plug_ip)
        except Exception:
            logger.exception("Failed to turn off smart plug at %s", self.plug_ip)

    def _schedule_off(self):
        """Schedule the light to turn off after the configured duration."""
        if self._off_timer is not None:
            self._off_timer.cancel()

        def _do_off():
            self._loop.run_until_complete(self._turn_off_light())

        self._off_timer = threading.Timer(self.light_duration, _do_off)
        self._off_timer.daemon = True
        self._off_timer.start()
        logger.info(
            "Light will turn off in %.0f seconds.", self.light_duration
        )
