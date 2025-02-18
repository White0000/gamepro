import time
import random
import math
import threading
import uuid
import collections
import sys
import os
import functools

from game_core.actions import ActionManager, ActionExecutionContext, ActionResultStatus
from game_core.rewards import RewardCalculator
from game_core.states import GameWorldState, GamePlayerState, GameEnemyState, GameItemState
from game_core.gameplay_manager import GameplayManager

class EnvironmentObservation:
    def __init__(self):
        self.player_position = (0, 0)
        self.player_health = 100
        self.player_mana = 100
        self.player_stamina = 100
        self.player_inventory = []
        self.enemies = []
        self.items = []
        self.current_area_id = None
        self.score = 0
        self.buff_states = []
        self.environment_variables = {}
        self.visible_map_data = []
        self.time_of_day = 0
        self.weather_state = ""
        self.buildings = []
        self.faction_relations = {}
        self.quests = []

class EnvironmentActionSpace:
    def __init__(self):
        self.available_actions = []
        self.discrete_actions = []
        self.continuous_actions = []
        self.multi_discrete_actions = []
        self.multi_continuous_actions = []

    def get_random_action(self):
        if self.discrete_actions:
            return random.choice(self.discrete_actions)
        return None

class EnvironmentConfig:
    def __init__(self):
        self.enable_multi_agent = False
        self.default_spawn_location = (0, 0)
        self.max_steps_per_episode = 500
        self.episode_timeout = 300.0
        self.seed = None
        self.enable_random_weather = True
        self.possible_weathers = ["Clear", "Rain", "Storm", "Fog", "Snow"]
        self.day_night_cycle_length = 120.0
        self.initial_time_of_day = 0.0
        self.enable_quests = True
        self.starting_quests = []
        self.enable_building_system = True
        self.max_buildings = 10
        self.allow_fast_travel = False
        self.enable_factions = True
        self.default_faction_relations = {}
        self.enable_score_system = True
        self.enable_buff_system = True
        self.enable_procedural_generation = False
        self.enable_tutorial = True
        self.tutorial_messages = []
        self.enable_pausing = True
        self.pause_on_focus_lost = False
        self.enable_difficulty_scaling = False
        self.difficulty_factor = 1.0
        self.enable_respawning_enemies = False
        self.respawn_interval = 60.0
        self.enable_npc_dialogues = True
        self.enable_crafting = True
        self.initial_resources = {}
        self.world_area_id = "default_area"
        self.enable_guild_system = False
        self.save_data_path = "environment_save.json"
        self.enable_logging = True

class EnvironmentLogger:
    def __init__(self, max_entries=10000):
        self.logs = collections.deque(maxlen=max_entries)

    def log(self, message):
        self.logs.append((time.time(), message))

    def get_logs(self):
        return list(self.logs)

class EnvironmentEvent:
    def __init__(self, event_id, event_type, timestamp, data=None):
        self.event_id = event_id
        self.event_type = event_type
        self.timestamp = timestamp
        self.data = data if data else {}

class EnvironmentEventManager:
    def __init__(self):
        self.subscribers = {}
        self.event_queue = collections.deque()
        self.lock = threading.Lock()

    def subscribe(self, event_type, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type, callback):
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)

    def emit_event(self, event_type, data=None):
        ev = EnvironmentEvent(uuid.uuid4(), event_type, time.time(), data)
        with self.lock:
            self.event_queue.append(ev)

    def process_events(self):
        with self.lock:
            while self.event_queue:
                ev = self.event_queue.popleft()
                if ev.event_type in self.subscribers:
                    for cb in self.subscribers[ev.event_type]:
                        cb(ev)

class MultiAgentController:
    def __init__(self):
        self.agents = {}

    def register_agent(self, agent_id, agent_state):
        self.agents[agent_id] = agent_state

    def remove_agent(self, agent_id):
        if agent_id in self.agents:
            del self.agents[agent_id]

    def get_agent_state(self, agent_id):
        return self.agents.get(agent_id, None)

    def reset_all_agents(self):
        for agent_id, st in self.agents.items():
            st.reset()

class EnvironmentTimeManager:
    def __init__(self):
        self.start_time = time.time()
        self.time_scale = 1.0
        self.paused = False
        self.paused_timestamp = 0.0
        self.total_paused_duration = 0.0

    def get_elapsed_time(self):
        if self.paused:
            return (self.paused_timestamp - self.start_time - self.total_paused_duration) * self.time_scale
        return (time.time() - self.start_time - self.total_paused_duration) * self.time_scale

    def pause(self):
        if not self.paused:
            self.paused = True
            self.paused_timestamp = time.time()

    def resume(self):
        if self.paused:
            now = time.time()
            self.total_paused_duration += (now - self.paused_timestamp)
            self.paused = False

class Environment:
    def __init__(self, config=None):
        self.config = config if config else EnvironmentConfig()
        self.world_state = GameWorldState()
        self.action_manager = ActionManager()
        self.multi_agent_controller = MultiAgentController()
        self.reward_calculator = RewardCalculator()
        self.gameplay_manager = GameplayManager()
        self.event_manager = EnvironmentEventManager()
        self.logger = EnvironmentLogger()
        self.env_time = EnvironmentTimeManager()
        self.action_space = EnvironmentActionSpace()
        self.observation_space = None
        self.episode_reward = 0.0
        self.episode_step_count = 0
        self.episode_start_time = 0.0
        self.done = False
        self.episode_id = uuid.uuid4()
        self.quest_data = {}
        self.buildings_data = []
        self.seed_value = self.config.seed if self.config.seed else int(time.time())
        random.seed(self.seed_value)
        self.last_reset_time = time.time()
        self.player_id = "Player0"
        self.init_environment()

    def init_environment(self):
        if self.config.enable_logging:
            self.logger.log("Environment initialization started.")
        self.world_state.world_area_id = self.config.world_area_id
        self.world_state.reset_world()
        if self.config.enable_multi_agent:
            self.multi_agent_controller.reset_all_agents()
        else:
            self.ensure_player_exists()
        self.setup_quests()
        self.setup_buildings()
        self.setup_action_space()
        self.episode_reward = 0.0
        self.episode_step_count = 0
        self.done = False
        self.episode_id = uuid.uuid4()
        self.episode_start_time = self.env_time.get_elapsed_time()
        if self.config.enable_random_weather:
            self.randomize_weather()
        self.world_state.time_of_day = self.config.initial_time_of_day
        if self.config.enable_logging:
            self.logger.log("Environment initialization completed.")

    def ensure_player_exists(self):
        player_state = self.world_state.get_player_state(self.player_id)
        if not player_state:
            p = GamePlayerState()
            p.player_id = self.player_id
            p.position = self.config.default_spawn_location
            p.hp = 100
            p.mana = 50
            p.stamina = 100
            self.world_state.register_player_state(self.player_id, p)

    def setup_quests(self):
        if self.config.enable_quests:
            for q in self.config.starting_quests:
                self.quest_data[q["quest_id"]] = {"status": "active", "progress": 0}

    def setup_buildings(self):
        if self.config.enable_building_system:
            self.buildings_data = []
            for i in range(self.config.max_buildings):
                self.buildings_data.append(None)

    def setup_action_space(self):
        self.action_space.available_actions = list(self.action_manager.actions.keys())
        self.action_space.discrete_actions = list(self.action_manager.actions.keys())

    def reset(self):
        self.init_environment()
        self.last_reset_time = time.time()
        obs = self.get_observation(self.player_id)
        return obs

    def step(self, action_id):
        if self.done:
            return self.get_observation(self.player_id), 0.0, True, {}
        self.episode_step_count += 1
        player_state = self.world_state.get_player_state(self.player_id)
        ctx = ActionExecutionContext()
        ctx.actor = player_state
        ctx.world = self.world_state
        ctx.position = player_state.position
        result = self.action_manager.trigger_action(action_id, ctx)
        self.world_state.update_entities()
        rew = self.reward_calculator.compute_reward(player_state, self.world_state, action_id, result)
        self.episode_reward += rew
        if result == ActionResultStatus.SUCCESS:
            self.process_gameplay_logic()
        done_cond = self.check_done_conditions()
        obs = self.get_observation(self.player_id)
        info = {}
        return obs, rew, done_cond, info

    def process_gameplay_logic(self):
        self.gameplay_manager.update_gameplay(self.world_state)
        self.update_time_of_day()
        if self.config.enable_npc_dialogues:
            pass
        if self.config.enable_respawning_enemies:
            pass
        if self.config.enable_factions:
            pass
        if self.config.enable_difficulty_scaling:
            pass
        self.event_manager.process_events()

    def check_done_conditions(self):
        if self.episode_step_count >= self.config.max_steps_per_episode:
            self.done = True
        if (self.env_time.get_elapsed_time() - self.episode_start_time) >= self.config.episode_timeout:
            self.done = True
        player_state = self.world_state.get_player_state(self.player_id)
        if player_state and player_state.hp <= 0:
            self.done = True
        return self.done

    def get_observation(self, agent_id):
        obs = EnvironmentObservation()
        st = self.world_state.get_player_state(agent_id)
        if st:
            obs.player_position = st.position
            obs.player_health = st.hp
            obs.player_mana = st.mana
            obs.player_stamina = st.stamina
            obs.player_inventory = list(st.inventory)
            obs.current_area_id = self.world_state.world_area_id
            obs.score = st.score
            obs.buff_states = list(st.buffs)
            obs.environment_variables["weather"] = self.world_state.weather_state
            obs.environment_variables["time_of_day"] = self.world_state.time_of_day
            obs.items = list(self.world_state.items.values())
            obs.enemies = list(self.world_state.enemies.values())
            obs.buildings = list(self.buildings_data)
            obs.faction_relations = dict(st.faction_relations)
            obs.quests = list(self.quest_data.items())
        return obs

    def randomize_weather(self):
        if self.config.enable_random_weather:
            w = random.choice(self.config.possible_weathers)
            self.world_state.weather_state = w

    def update_time_of_day(self):
        day_cycle_length = self.config.day_night_cycle_length
        current_time = self.world_state.time_of_day
        elapsed = self.env_time.get_elapsed_time() - self.episode_start_time
        cycle_progress = (elapsed % day_cycle_length) / day_cycle_length
        self.world_state.time_of_day = cycle_progress * 24.0

    def close(self):
        if self.config.enable_logging:
            self.logger.log("Environment closed.")

    def get_logs(self):
        return self.logger.get_logs()

    def seed(self, seed_value):
        self.seed_value = seed_value
        random.seed(self.seed_value)

    def render(self, mode="human"):
        pass

    def pause(self):
        if self.config.enable_pausing:
            self.env_time.pause()

    def resume(self):
        if self.config.enable_pausing:
            self.env_time.resume()

    def manual_save(self):
        pass

    def manual_load(self):
        pass

    def multi_agent_step(self, actions_dict):
        results = {}
        if not self.config.enable_multi_agent:
            return {}
        for agent_id, action_id in actions_dict.items():
            agent_state = self.world_state.get_player_state(agent_id)
            if not agent_state:
                continue
            ctx = ActionExecutionContext()
            ctx.actor = agent_state
            ctx.world = self.world_state
            ctx.position = agent_state.position
            res = self.action_manager.trigger_action(action_id, ctx)
            r = self.reward_calculator.compute_reward(agent_state, self.world_state, action_id, res)
            results[agent_id] = r
        self.world_state.update_entities()
        self.process_gameplay_logic()
        done_cond = self.check_done_conditions()
        obs = {}
        for agent_id in actions_dict:
            obs[agent_id] = self.get_observation(agent_id)
        info = {}
        return obs, results, done_cond, info

    def multi_agent_reset(self):
        self.init_environment()
        obs = {}
        for agent_id in self.multi_agent_controller.agents.keys():
            obs[agent_id] = self.get_observation(agent_id)
        return obs

    def add_event_listener(self, event_type, callback):
        self.event_manager.subscribe(event_type, callback)

    def remove_event_listener(self, event_type, callback):
        self.event_manager.unsubscribe(event_type, callback)

    def emit_custom_event(self, event_type, data=None):
        self.event_manager.emit_event(event_type, data)

    def get_time(self):
        return self.env_time.get_elapsed_time()

    def get_episode_reward(self):
        return self.episode_reward

    def get_current_step(self):
        return self.episode_step_count

    def is_done(self):
        return self.done
