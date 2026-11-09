"""Cumulative client-observed LLM/agent planning time, excluding host tool work."""
import datetime
import json
import os
from pathlib import Path
import signal
import time

PLANNING_LIMIT_SECONDS = 1800


class PlanningTimeExhausted(TimeoutError):
    pass


def used_seconds(snapshot, now=None):
    value = float(snapshot.get('used_seconds', 0))
    start = snapshot.get('active_since_monotonic')
    if start is not None and not snapshot.get('closed'):
        value += max(0., (time.monotonic() if now is None else now)-float(start))
    return value


class PlanningBudget:
    def __init__(self, path, limit=PLANNING_LIMIT_SECONDS, clock=time.monotonic, alarms=True):
        self.path = Path(path)
        self.limit, self.clock, self.alarms = float(limit), clock, alarms
        self.used = 0.
        self.active_since = None
        self.closed = self.exhausted = False
        self.phase = 'initializing_model'
        self.reason = None
        self.previous_handler = None
        self._publish()

    def snapshot(self):
        return dict(version='cumulative_planning_v1', limit_seconds=self.limit,
            used_seconds=self.used, active_since_monotonic=self.active_since,
            closed=self.closed, exhausted=self.exhausted, phase=self.phase,
            stop_reason=self.reason, updated_monotonic=self.clock(),
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            includes='API queue/network/retries, inference, context compaction and in-memory agent planning',
            excludes='initialization, host tools, simulation/rendering, native-success hold and artifact finalization')

    def _publish(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name+f'.{os.getpid()}.{time.monotonic_ns()}.tmp')
        temp.write_text(json.dumps(self.snapshot(), indent=2)+'\n')
        temp.replace(self.path)

    def _disarm(self):
        if self.alarms:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def _expire(self, *args):
        self._disarm()
        if self.active_since is not None:
            self.used += max(0., self.clock()-self.active_since)
        self.active_since = None
        self.exhausted = True
        self.phase = 'planning_time_exhausted'
        self.reason = 'planning_time_exhausted'
        self._publish()
        raise PlanningTimeExhausted('Cumulative LLM planning budget exhausted')

    def resume(self):
        if self.closed:
            raise RuntimeError('Cannot resume planning after the model session was closed')
        if self.exhausted or self.used >= self.limit:
            self._expire()
        if self.active_since is not None:
            return  # Retries and nested turn starts must not reset the clock.
        self.active_since = self.clock()
        self.phase = 'planning'
        self._publish()
        if self.alarms:
            if self.previous_handler is None:
                self.previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._expire)
            remaining = self.limit-used_seconds(self.snapshot(), self.clock())
            if remaining <= 0:
                self._expire()
            signal.setitimer(signal.ITIMER_REAL, remaining)

    def pause(self, phase='host_tool', check_limit=True):
        self._disarm()
        if self.active_since is not None:
            self.used += max(0., self.clock()-self.active_since)
            self.active_since = None
        self.phase = phase
        if check_limit and self.used >= self.limit:
            self._expire()
        self._publish()

    def close(self, reason):
        self.pause('model_closed', check_limit=False)
        self.closed = True
        self.reason = reason
        self._publish()
        if self.alarms and self.previous_handler is not None:
            signal.signal(signal.SIGALRM, self.previous_handler)
