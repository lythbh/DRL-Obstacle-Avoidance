"""Curriculum manager for progressive difficulty training."""
from dataclasses import dataclass


CONSECUTIVE_GOALS_TO_ADVANCE = 10
MAX_STAGE_EPISODES = 2500


@dataclass
class StageConfig:
    num_obstacles: int
    goal_y_range: float       # 0.0 = fixed goal at default position
    start_yaw_noise: float
    start_position_noise: float
    description: str


CURRICULUM: list = [
    StageConfig(
        num_obstacles=0,
        goal_y_range=0.0,
        start_yaw_noise=0.2,
        start_position_noise=0.04,
        description="Stage 1: empty world, fixed goal — learn basic goal-seeking",
    ),
    StageConfig(
        num_obstacles=5,
        goal_y_range=0.7,
        start_yaw_noise=0.5,
        start_position_noise=0.06,
        description="Stage 2: sparse obstacles, moderate goal randomization",
    ),
    StageConfig(
        num_obstacles=15,
        goal_y_range=1.5,
        start_yaw_noise=0.8,
        start_position_noise=0.08,
        description="Stage 3: full randomization — target distribution",
    ),
]


class CurriculumManager:
    """Tracks training stage and advances when mastery criteria are met.

    Advances to next stage when the agent hits the goal 10 times in a row
    OR when the current stage has run for 2500 episodes — whichever comes first.
    Stays at the final stage once reached.
    """

    def __init__(self) -> None:
        self._stage_idx: int = 0
        self._consecutive_goals: int = 0
        self._stage_episodes: int = 0

    @property
    def stage(self) -> StageConfig:
        return CURRICULUM[self._stage_idx]

    @property
    def stage_num(self) -> int:
        return self._stage_idx + 1

    @property
    def num_stages(self) -> int:
        return len(CURRICULUM)

    @property
    def at_final_stage(self) -> bool:
        return self._stage_idx >= len(CURRICULUM) - 1

    @property
    def consecutive_goals(self) -> int:
        return self._consecutive_goals

    @property
    def stage_episodes(self) -> int:
        return self._stage_episodes

    def on_episode_end(self, goal_reached: bool) -> bool:
        """Update state after an episode. Returns True if stage advanced."""
        self._stage_episodes += 1
        if goal_reached:
            self._consecutive_goals += 1
        else:
            self._consecutive_goals = 0

        if self.at_final_stage:
            return False

        advanced = (
            self._consecutive_goals >= CONSECUTIVE_GOALS_TO_ADVANCE
            or self._stage_episodes >= MAX_STAGE_EPISODES
        )
        if advanced:
            self._stage_idx += 1
            self._consecutive_goals = 0
            self._stage_episodes = 0
            return True
        return False

    def status_str(self) -> str:
        return (
            f"stage={self.stage_num}/{self.num_stages} "
            f"streak={self._consecutive_goals}/{CONSECUTIVE_GOALS_TO_ADVANCE} "
            f"stage_ep={self._stage_episodes}"
        )
