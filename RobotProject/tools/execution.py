from dataclasses import dataclass

from tools.encode import parse_action


@dataclass(frozen=True)
class ActionResult:
    success: bool
    message: str


class MockRobotExecutor:
    """Replace this class with ROS/service calls without changing the planner."""

    def __init__(self, fail_once=None):
        self.fail_once = set(fail_once or [])
        self.failed = set()

    def execute(self, action, observation):
        verb, argument = parse_action(action)
        failure_key = action if action in self.fail_once else verb
        if failure_key in self.fail_once and failure_key not in self.failed:
            self.failed.add(failure_key)
            return ActionResult(False, "simulated transient failure")

        if verb in {"locate", "grasp"} and not observation.contains(argument):
            return ActionResult(False, f"{argument} is not visible")
        return ActionResult(True, "completed")

