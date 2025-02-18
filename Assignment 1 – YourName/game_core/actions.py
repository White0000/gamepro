import enum
import math
import random
import time
import uuid
import sys
import threading
import collections

class ActionCategory(enum.Enum):
    MOVEMENT = 1
    COMBAT = 2
    INTERACTION = 3
    INVENTORY = 4
    SPECIAL = 5
    BUILDING = 6
    SYSTEM = 7
    DEBUG = 8
    CUSTOM = 9

class ActionResultStatus(enum.Enum):
    SUCCESS = 1
    FAILURE = 2
    PENDING = 3
    INTERRUPTED = 4
    INVALID = 5

class ActionKeyBinding:
    def __init__(self, action_id, key_code, modifier=None, device="keyboard"):
        self.action_id = action_id
        self.key_code = key_code
        self.modifier = modifier
        self.device = device

class ActionExecutionContext:
    def __init__(self):
        self.actor = None
        self.target = None
        self.world = None
        self.position = None
        self.timestamp = time.time()
        self.additional_data = {}
        self.flags = set()

class ActionEnergyRequirement:
    def __init__(self, energy_cost=0, resource_type="stamina"):
        self.energy_cost = energy_cost
        self.resource_type = resource_type

class ActionCooldown:
    def __init__(self, cooldown_time=0.0):
        self.cooldown_time = cooldown_time
        self.last_use_timestamp = 0.0

    def is_on_cooldown(self):
        now = time.time()
        return (now - self.last_use_timestamp) < self.cooldown_time

    def remaining_time(self):
        now = time.time()
        remaining = self.cooldown_time - (now - self.last_use_timestamp)
        if remaining < 0:
            remaining = 0
        return remaining

    def trigger(self):
        self.last_use_timestamp = time.time()

class ActionRequirement:
    def __init__(self, min_level=1, required_items=None, restricted_areas=None, faction_reputation=None):
        self.min_level = min_level
        self.required_items = required_items if required_items else []
        self.restricted_areas = restricted_areas if restricted_areas else []
        self.faction_reputation = faction_reputation if faction_reputation else {}

class ActionEffect:
    def __init__(self, health_delta=0, stamina_delta=0, mana_delta=0, buff_id=None, buff_duration=0.0):
        self.health_delta = health_delta
        self.stamina_delta = stamina_delta
        self.mana_delta = mana_delta
        self.buff_id = buff_id
        self.buff_duration = buff_duration

class ComboStep:
    def __init__(self, action_id, window_start, window_end):
        self.action_id = action_id
        self.window_start = window_start
        self.window_end = window_end

class ActionCombo:
    def __init__(self, combo_id, steps=None, final_action_id=None):
        self.combo_id = combo_id
        self.steps = steps if steps else []
        self.final_action_id = final_action_id
        self.current_step_index = 0
        self.start_time = None

    def reset(self):
        self.current_step_index = 0
        self.start_time = None

    def advance(self, action_id, elapsed):
        if self.current_step_index >= len(self.steps):
            return False
        step = self.steps[self.current_step_index]
        if step.action_id == action_id and step.window_start <= elapsed <= step.window_end:
            self.current_step_index += 1
            return True
        return False

    def is_complete(self):
        return self.current_step_index == len(self.steps)

class Action:
    def __init__(self, action_id, name, category=ActionCategory.MOVEMENT):
        self.action_id = action_id
        self.name = name
        self.category = category
        self.energy_requirement = ActionEnergyRequirement()
        self.cooldown = ActionCooldown()
        self.requirement = ActionRequirement()
        self.effects = []
        self.icon_path = None
        self.animation_id = None
        self.use_sound = None
        self.cast_time = 0.0
        self.is_channeling = False
        self.channel_duration = 0.0
        self.range = 1.0
        self.aoe_radius = 0.0
        self.target_type = "none"
        self.auto_trigger = False
        self.script_id = None
        self.interruptible = True
        self.disables_movement = False
        self.disables_combat = False
        self.ai_usable = True
        self.priority = 1
        self.base_damage = 0
        self.damage_type = "physical"
        self.scaling_stat = None
        self.scaling_factor = 1.0
        self.projectile_speed = 0.0
        self.projectile_gravity = 0.0
        self.projectile_lifetime = 0.0
        self.piercing = False
        self.knockback_force = 0.0
        self.knockback_angle = 0.0
        self.has_global_cooldown = False
        self.global_cooldown_time = 0.0
        self.tags = []
        self.is_toggle = False
        self.toggle_state = False
        self.toggle_off_action_id = None
        self.combo_chain_id = None
        self.cancelable_by_user = False
        self.can_crit = False
        self.crit_chance = 0.0
        self.crit_multiplier = 2.0
        self.life_steal_percent = 0.0
        self.mana_steal_percent = 0.0
        self.shield_penetration = 0.0
        self.on_hit_script_id = None
        self.summon_entity_id = None
        self.summon_count = 0
        self.teleport_destination = None
        self.temporary_attribute_mods = {}
        self.required_stance = None
        self.forbidden_stance = None
        self.overrides_stance = None
        self.collect_item_id = None
        self.collect_item_count = 0
        self.consume_on_use = False
        self.place_building_id = None
        self.building_upgrade_id = None
        self.system_command = None
        self.debug_info = None
        self.xp_gain = 0
        self.reputation_gain = 0
        self.channel_tick_rate = 0.0
        self.channel_tick_script = None
        self.end_channel_script = None
        self.station_required_id = None
        self.recipe_id = None
        self.cast_cancel_on_move = False
        self.cast_cancel_on_damage = False
        self.auto_target_friendly = False
        self.auto_target_enemy = False
        self.custom_validation_method = None

    def can_execute(self, ctx: ActionExecutionContext):
        if self.cooldown.is_on_cooldown():
            return False
        if ctx.actor and ctx.actor.level < self.requirement.min_level:
            return False
        for item_id in self.requirement.required_items:
            if not ctx.actor or not ctx.actor.has_item(item_id):
                return False
        if ctx.actor and ctx.actor.energy < self.energy_requirement.energy_cost:
            return False
        if self.requirement.restricted_areas:
            if ctx.world and ctx.world.current_area_id in self.requirement.restricted_areas:
                return False
        for faction, rep_req in self.requirement.faction_reputation.items():
            if not ctx.actor or ctx.actor.get_faction_reputation(faction) < rep_req:
                return False
        if self.custom_validation_method:
            if not self.custom_validation_method(ctx, self):
                return False
        return True

    def apply_cooldown(self):
        self.cooldown.trigger()

    def execute(self, ctx: ActionExecutionContext):
        result = ActionResultStatus.INVALID
        if not self.can_execute(ctx):
            result = ActionResultStatus.FAILURE
            return result
        if self.cast_time > 0 and not self.is_channeling:
            result = ActionResultStatus.PENDING
            return result
        self.apply_cooldown()
        if ctx.actor:
            ctx.actor.energy -= self.energy_requirement.energy_cost
        self.apply_effects(ctx)
        if self.is_toggle:
            self.toggle_state = not self.toggle_state
        result = ActionResultStatus.SUCCESS
        return result

    def apply_effects(self, ctx: ActionExecutionContext):
        for eff in self.effects:
            if ctx.actor:
                ctx.actor.hp += eff.health_delta
                ctx.actor.stamina += eff.stamina_delta
                ctx.actor.mana += eff.mana_delta

    def interrupt(self):
        pass

class ActionValidator:
    def __init__(self):
        self.error_code = None
        self.error_message = None

    def validate_action(self, action: Action, ctx: ActionExecutionContext):
        valid = action.can_execute(ctx)
        if not valid:
            self.error_code = "INVALID_ACTION"
            self.error_message = "Cannot execute action"
        else:
            self.error_code = None
            self.error_message = None
        return valid

class ActionQueueItem:
    def __init__(self, action: Action, timestamp):
        self.action = action
        self.timestamp = timestamp

class ActionQueue:
    def __init__(self):
        self.queue = collections.deque()
        self.lock = threading.Lock()

    def enqueue(self, action: Action):
        with self.lock:
            self.queue.append(ActionQueueItem(action, time.time()))

    def dequeue(self):
        with self.lock:
            if self.queue:
                return self.queue.popleft()
        return None

    def clear(self):
        with self.lock:
            self.queue.clear()

class ActionPipeline:
    def __init__(self):
        self.current_action = None
        self.next_action = None
        self.action_start_time = None
        self.action_end_time = None
        self.on_complete_callback = None

    def start_action(self, action: Action, ctx: ActionExecutionContext):
        self.current_action = action
        self.action_start_time = time.time()
        if action.cast_time > 0 and not action.is_channeling:
            self.action_end_time = self.action_start_time + action.cast_time
        else:
            self.action_end_time = self.action_start_time
        return action.execute(ctx)

    def update(self, ctx: ActionExecutionContext):
        if self.current_action and self.current_action.cast_time > 0 and not self.current_action.is_channeling:
            now = time.time()
            if now >= self.action_end_time:
                status = self.current_action.execute(ctx)
                if self.on_complete_callback:
                    self.on_complete_callback(self.current_action, status)
                self.current_action = None
        elif self.current_action and self.current_action.is_channeling:
            now = time.time()
            if now >= self.action_end_time:
                status = self.current_action.execute(ctx)
                if self.on_complete_callback:
                    self.on_complete_callback(self.current_action, status)
                self.current_action = None

class ActionBindingManager:
    def __init__(self):
        self.bindings = []

    def bind_key(self, action_id, key_code, modifier=None, device="keyboard"):
        self.bindings.append(ActionKeyBinding(action_id, key_code, modifier, device))

    def get_binding(self, action_id):
        for b in self.bindings:
            if b.action_id == action_id:
                return b
        return None

class ActionManager:
    def __init__(self):
        self.actions = {}
        self.combos = {}
        self.active_combo = None
        self.active_combo_start_time = 0.0
        self.global_cooldown = ActionCooldown()
        self.global_cooldown_active = False
        self.global_cooldown_duration = 0.0

    def register_action(self, action: Action):
        self.actions[action.action_id] = action

    def register_combo(self, combo: ActionCombo):
        self.combos[combo.combo_id] = combo

    def find_action(self, action_id):
        return self.actions.get(action_id, None)

    def trigger_action(self, action_id, ctx: ActionExecutionContext):
        if self.global_cooldown_active:
            if self.global_cooldown.is_on_cooldown():
                return ActionResultStatus.FAILURE
            else:
                self.global_cooldown_active = False
        action = self.find_action(action_id)
        if not action:
            return ActionResultStatus.INVALID
        if action.has_global_cooldown:
            self.global_cooldown_duration = action.global_cooldown_time
        status = action.execute(ctx)
        if status == ActionResultStatus.SUCCESS and action.has_global_cooldown:
            self.global_cooldown.cooldown_time = self.global_cooldown_duration
            self.global_cooldown.trigger()
            self.global_cooldown_active = True
        return status

    def start_combo(self, combo_id):
        combo = self.combos.get(combo_id, None)
        if combo:
            combo.reset()
            self.active_combo = combo
            self.active_combo_start_time = time.time()

    def progress_combo(self, action_id):
        if not self.active_combo:
            return False
        elapsed = time.time() - self.active_combo_start_time
        if self.active_combo.advance(action_id, elapsed):
            if self.active_combo.is_complete():
                final_action = self.find_action(self.active_combo.final_action_id)
                self.active_combo.reset()
                self.active_combo = None
                return True if final_action else False
            return True
        else:
            self.active_combo.reset()
            self.active_combo = None
            return False

    def clear_actions(self):
        self.actions.clear()

    def clear_combos(self):
        self.combos.clear()

class ActionDebugTools:
    def __init__(self, action_manager: ActionManager):
        self.action_manager = action_manager

    def dump_action_list(self):
        data = []
        for a_id, act in self.action_manager.actions.items():
            data.append(a_id)
        return data

    def dump_action_details(self, action_id):
        act = self.action_manager.find_action(action_id)
        if not act:
            return None
        return {
            "id": act.action_id,
            "name": act.name,
            "category": act.category.name,
            "energy_cost": act.energy_requirement.energy_cost,
            "cooldown": act.cooldown.cooldown_time,
            "cast_time": act.cast_time,
            "is_channeling": act.is_channeling,
            "channel_duration": act.channel_duration,
            "range": act.range,
            "aoe_radius": act.aoe_radius,
            "tags": act.tags
        }

    def debug_trigger_action(self, action_id, ctx: ActionExecutionContext):
        return self.action_manager.trigger_action(action_id, ctx)

    def debug_combo(self, combo_id, sequence):
        self.action_manager.start_combo(combo_id)
        success_count = 0
        for sid in sequence:
            res = self.action_manager.progress_combo(sid)
            if res:
                success_count += 1
        return success_count

class ActionLogger:
    def __init__(self):
        self.logs = collections.deque(maxlen=1000)

    def log_action(self, action: Action, status: ActionResultStatus):
        self.logs.append((time.time(), action.action_id, action.name, status.name))

    def get_logs(self):
        return list(self.logs)

class ActionScheduler:
    def __init__(self):
        self.scheduled_actions = []
        self.lock = threading.Lock()

    def schedule_action(self, action: Action, ctx: ActionExecutionContext, execute_time):
        with self.lock:
            self.scheduled_actions.append((execute_time, action, ctx))

    def update(self):
        now = time.time()
        fired = []
        with self.lock:
            remain = []
            for (t, a, c) in self.scheduled_actions:
                if now >= t:
                    fired.append((a, c))
                else:
                    remain.append((t, a, c))
            self.scheduled_actions = remain
        for (action, ctx) in fired:
            action.execute(ctx)
