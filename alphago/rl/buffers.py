import numpy as np


class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []

    def record_decision(self, state, action, estimated_value=0):
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimated_values.append(estimated_value)

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - self._current_episode_estimated_values[i]
            self.advantages.append(advantage)

        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimated_values = []


class ExperienceBuffer:
    def __init__(self, states, actions, rewards, advantages):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages

    def save_experience(self, path):
        """Save experiences as npz"""
        np.savez(
            path,
            states=self.states,
            actions=self.actions,
            rewards=self.rewards,
            advantages=self.advantages,
        )

    @classmethod
    def load_experience(cls, path):
        """Load experiences"""
        data = np.load(path)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        advantages = data["advantages"]
        return cls(states, actions, rewards, advantages)

    @classmethod
    def combine_buffers(cls, buffers):
        """Combine multiple buffers into one"""
        states = np.concatenate([buffer.states for buffer in buffers])
        actions = np.concatenate([buffer.actions for buffer in buffers])
        rewards = np.concatenate([buffer.rewards for buffer in buffers])
        advantages = np.concatenate([buffer.advantages for buffer in buffers])
        return cls(states, actions, rewards, advantages)
