import os
import sys
import json
import random
import time
import math
import shutil
import pathlib
import subprocess
import platform
import secrets
import uuid
import threading

class RoomConfig:
    def __init__(self):
        self.room_id = 0
        self.width = 10
        self.height = 10
        self.theme = "default"
        self.floor_texture = "floor.png"
        self.wall_texture = "wall.png"
        self.enemy_count = 0
        self.enemy_types = []
        self.item_count = 0
        self.item_types = []
        self.has_boss = False
        self.boss_id = None
        self.spawn_chance = 1.0
        self.min_player_level = 1
        self.environment_effects = []
        self.lighting_type = "standard"
        self.is_safe_zone = False
        self.scripted_event_id = None
        self.event_trigger_chance = 0.0
        self.secret_rooms = 0
        self.secret_room_chance = 0.0
        self.secret_room_types = []
        self.loot_multiplier = 1.0
        self.difficulty_scale = 1.0
        self.allow_resurgence = False
        self.resurgence_cooldown = 0
        self.ambient_sound = ""
        self.background_music = ""
        self.background_music_volume = 0.8
        self.disable_minimap = False
        self.override_minimap_color = None
        self.override_minimap_icon = None
        self.enable_puzzles = False
        self.puzzle_count = 0
        self.puzzle_types = []
        self.trap_count = 0
        self.trap_types = []
        self.npc_count = 0
        self.npc_behaviors = []
        self.room_tag = ""
        self.max_enemies_per_wave = 0
        self.wave_spawn_interval = 10
        self.allow_random_boss_spawn = False
        self.random_boss_chance = 0.0
        self.fog_density = 0.0
        self.fog_color = "#000000"
        self.gravity_override = 1.0
        self.physics_material = None
        self.allow_climbing = False
        self.locked = False
        self.lock_key_id = None
        self.auto_unlock_on_clear = False
        self.reward_chest_count = 0
        self.reward_chest_types = []
        self.environment_type = "indoor"
        self.decoration_count = 0
        self.decoration_prefabs = []
        self.prevent_fast_travel = False
        self.ambient_light_color = "#FFFFFF"
        self.ambient_light_intensity = 1.0
        self.post_process_profile = ""
        self.enemy_speed_multiplier = 1.0
        self.enemy_health_multiplier = 1.0
        self.enemy_attack_multiplier = 1.0
        self.allow_companion_spawn = False
        self.companion_spawn_chance = 0.0
        self.room_connectors = []
        self.room_script_path = None
        self.terrain_variation = 0
        self.spawn_points = []
        self.item_spots = []
        self.trap_spots = []
        self.decoration_spots = []
        self.wave_count = 0
        self.wave_enemy_list = []
        self.wave_boss_list = []
        self.special_objectives = []
        self.loot_drop_rate = 1.0
        self.enable_random_events = False
        self.random_event_chance = 0.0
        self.random_event_types = []
        self.theme_music_cue = ""
        self.ai_alert_radius_multiplier = 1.0
        self.ambient_particle_effect = ""
        self.level_requirement_override = None
        self.custom_room_logic = False
        self.custom_logic_path = ""
        self.max_players_allowed = 1
        self.enable_coop_triggers = False
        self.spawner_limit = 0
        self.generation_weight = 1.0
        self.enable_portals = False
        self.portal_count = 0
        self.portal_destinations = []
        self.room_visibility_range = 50.0
        self.lockdown_on_entry = False
        self.lockdown_duration = 0
        self.lockdown_release_condition = ""
        self.allow_air_dashes = True
        self.allow_wall_jumps = False
        self.surface_friction = 1.0
        self.bgm_fade_in_time = 0.0
        self.bgm_fade_out_time = 0.0
        self.has_water_zone = False
        self.water_level = 0
        self.disable_player_abilities = []
        self.allow_stealth_mechanics = True
        self.interactable_objects = []
        self.randomize_lights = False
        self.light_color_options = []
        self.light_intensity_min = 0.5
        self.light_intensity_max = 1.5
        self.loot_table_id = None
        self.texture_variants = []
        self.collision_map_path = ""
        self.enable_vertical_levels = False
        self.vertical_section_count = 0
        self.randomize_enemies = False
        self.random_enemy_pool = []
        self.random_enemy_chance = 0.0
        self.post_room_script = ""
        self.ai_spawn_delay = 0.0
        self.death_zone_height = -9999
        self.powerup_spot_count = 0
        self.powerup_spot_chance = 0.0
        self.powerup_types = []
        self.random_room_attachments = []
        self.custom_room_data = {}
        self.room_seed = None
        self.room_seed_overrides_random = False
        self.minimap_display_name = ""
        self.minimap_color_override = None
        self.minimap_icon_override = None
        self.access_requirements = []
        self.procedural_lock_chance = 0.0
        self.lock_destruction_allowed = False
        self.transition_cinematic_id = None
        self.player_stat_modifiers = {}
        self.enable_bonfires = False
        self.bonfire_count = 0
        self.bonfire_spots = []
        self.allows_mounts = True
        self.requires_underwater_gear = False
        self.underwater_gear_id = None
        self.enable_timed_events = False
        self.timed_event_duration = 0
        self.bgm_loop_point = 0.0
        self.parallax_layers = []
        self.enable_parallax = False
        self.exploration_bonus = 0
        self.item_respawn_time = 0
        self.environment_performance_cost = 1
        self.supports_night_vision = False
        self.night_vision_filter = ""
        self.god_rays_enabled = False
        self.god_rays_intensity = 1.0
        self.destruction_physics = False
        self.destruction_debris_duration = 0
        self.scripts_on_clear = []
        self.scripts_on_enter = []
        self.scripts_on_exit = []
        self.room_transformation_id = None
        self.dynamic_door_states = {}
        self.ally_spawn_points = []
        self.ally_types = []
        self.ally_chance = 0.0
        self.power_required = 0
        self.power_outlets = []
        self.mana_crystals_available = 0
        self.alternate_gravity_zones = []
        self.minimum_wave_interval = 5
        self.maximum_wave_interval = 60
        self.enable_wave_scaling = True
        self.wave_scaling_factor = 1.05
        self.wave_scaling_limit = 10
        self.wave_infinite_mode = False
        self.cinematic_sequences = []
        self.collectibles_count = 0
        self.collectibles_types = []
        self.enforce_level_sync = False
        self.level_sync_item = None
        self.trigger_volume_count = 0
        self.trigger_volume_list = []
        self.deactivate_on_completion = False
        self.completion_award_id = None
        self.completion_skip_allowed = False
        self.linked_rooms = []
        self.force_single_player = False
        self.has_side_quests = False
        self.side_quest_ids = []
        self.can_build_structures = False
        self.buildable_structure_ids = []
        self.enable_clan_system = False
        self.clan_ownership_required = False
        self.default_spawn_region = None

    def to_dict(self):
        return {
            "room_id": self.room_id,
            "width": self.width,
            "height": self.height,
            "theme": self.theme,
            "floor_texture": self.floor_texture,
            "wall_texture": self.wall_texture,
            "enemy_count": self.enemy_count,
            "enemy_types": self.enemy_types,
            "item_count": self.item_count,
            "item_types": self.item_types,
            "has_boss": self.has_boss,
            "boss_id": self.boss_id,
            "spawn_chance": self.spawn_chance,
            "min_player_level": self.min_player_level,
            "environment_effects": self.environment_effects,
            "lighting_type": self.lighting_type,
            "is_safe_zone": self.is_safe_zone,
            "scripted_event_id": self.scripted_event_id,
            "event_trigger_chance": self.event_trigger_chance,
            "secret_rooms": self.secret_rooms,
            "secret_room_chance": self.secret_room_chance,
            "secret_room_types": self.secret_room_types,
            "loot_multiplier": self.loot_multiplier,
            "difficulty_scale": self.difficulty_scale,
            "allow_resurgence": self.allow_resurgence,
            "resurgence_cooldown": self.resurgence_cooldown,
            "ambient_sound": self.ambient_sound,
            "background_music": self.background_music,
            "background_music_volume": self.background_music_volume,
            "disable_minimap": self.disable_minimap,
            "override_minimap_color": self.override_minimap_color,
            "override_minimap_icon": self.override_minimap_icon,
            "enable_puzzles": self.enable_puzzles,
            "puzzle_count": self.puzzle_count,
            "puzzle_types": self.puzzle_types,
            "trap_count": self.trap_count,
            "trap_types": self.trap_types,
            "npc_count": self.npc_count,
            "npc_behaviors": self.npc_behaviors,
            "room_tag": self.room_tag,
            "max_enemies_per_wave": self.max_enemies_per_wave,
            "wave_spawn_interval": self.wave_spawn_interval,
            "allow_random_boss_spawn": self.allow_random_boss_spawn,
            "random_boss_chance": self.random_boss_chance,
            "fog_density": self.fog_density,
            "fog_color": self.fog_color,
            "gravity_override": self.gravity_override,
            "physics_material": self.physics_material,
            "allow_climbing": self.allow_climbing,
            "locked": self.locked,
            "lock_key_id": self.lock_key_id,
            "auto_unlock_on_clear": self.auto_unlock_on_clear,
            "reward_chest_count": self.reward_chest_count,
            "reward_chest_types": self.reward_chest_types,
            "environment_type": self.environment_type,
            "decoration_count": self.decoration_count,
            "decoration_prefabs": self.decoration_prefabs,
            "prevent_fast_travel": self.prevent_fast_travel,
            "ambient_light_color": self.ambient_light_color,
            "ambient_light_intensity": self.ambient_light_intensity,
            "post_process_profile": self.post_process_profile,
            "enemy_speed_multiplier": self.enemy_speed_multiplier,
            "enemy_health_multiplier": self.enemy_health_multiplier,
            "enemy_attack_multiplier": self.enemy_attack_multiplier,
            "allow_companion_spawn": self.allow_companion_spawn,
            "companion_spawn_chance": self.companion_spawn_chance,
            "room_connectors": self.room_connectors,
            "room_script_path": self.room_script_path,
            "terrain_variation": self.terrain_variation,
            "spawn_points": self.spawn_points,
            "item_spots": self.item_spots,
            "trap_spots": self.trap_spots,
            "decoration_spots": self.decoration_spots,
            "wave_count": self.wave_count,
            "wave_enemy_list": self.wave_enemy_list,
            "wave_boss_list": self.wave_boss_list,
            "special_objectives": self.special_objectives,
            "loot_drop_rate": self.loot_drop_rate,
            "enable_random_events": self.enable_random_events,
            "random_event_chance": self.random_event_chance,
            "random_event_types": self.random_event_types,
            "theme_music_cue": self.theme_music_cue,
            "ai_alert_radius_multiplier": self.ai_alert_radius_multiplier,
            "ambient_particle_effect": self.ambient_particle_effect,
            "level_requirement_override": self.level_requirement_override,
            "custom_room_logic": self.custom_room_logic,
            "custom_logic_path": self.custom_logic_path,
            "max_players_allowed": self.max_players_allowed,
            "enable_coop_triggers": self.enable_coop_triggers,
            "spawner_limit": self.spawner_limit,
            "generation_weight": self.generation_weight,
            "enable_portals": self.enable_portals,
            "portal_count": self.portal_count,
            "portal_destinations": self.portal_destinations,
            "room_visibility_range": self.room_visibility_range,
            "lockdown_on_entry": self.lockdown_on_entry,
            "lockdown_duration": self.lockdown_duration,
            "lockdown_release_condition": self.lockdown_release_condition,
            "allow_air_dashes": self.allow_air_dashes,
            "allow_wall_jumps": self.allow_wall_jumps,
            "surface_friction": self.surface_friction,
            "bgm_fade_in_time": self.bgm_fade_in_time,
            "bgm_fade_out_time": self.bgm_fade_out_time,
            "has_water_zone": self.has_water_zone,
            "water_level": self.water_level,
            "disable_player_abilities": self.disable_player_abilities,
            "allow_stealth_mechanics": self.allow_stealth_mechanics,
            "interactable_objects": self.interactable_objects,
            "randomize_lights": self.randomize_lights,
            "light_color_options": self.light_color_options,
            "light_intensity_min": self.light_intensity_min,
            "light_intensity_max": self.light_intensity_max,
            "loot_table_id": self.loot_table_id,
            "texture_variants": self.texture_variants,
            "collision_map_path": self.collision_map_path,
            "enable_vertical_levels": self.enable_vertical_levels,
            "vertical_section_count": self.vertical_section_count,
            "randomize_enemies": self.randomize_enemies,
            "random_enemy_pool": self.random_enemy_pool,
            "random_enemy_chance": self.random_enemy_chance,
            "post_room_script": self.post_room_script,
            "ai_spawn_delay": self.ai_spawn_delay,
            "death_zone_height": self.death_zone_height,
            "powerup_spot_count": self.powerup_spot_count,
            "powerup_spot_chance": self.powerup_spot_chance,
            "powerup_types": self.powerup_types,
            "random_room_attachments": self.random_room_attachments,
            "custom_room_data": self.custom_room_data,
            "room_seed": self.room_seed,
            "room_seed_overrides_random": self.room_seed_overrides_random,
            "minimap_display_name": self.minimap_display_name,
            "minimap_color_override": self.minimap_color_override,
            "minimap_icon_override": self.minimap_icon_override,
            "access_requirements": self.access_requirements,
            "procedural_lock_chance": self.procedural_lock_chance,
            "lock_destruction_allowed": self.lock_destruction_allowed,
            "transition_cinematic_id": self.transition_cinematic_id,
            "player_stat_modifiers": self.player_stat_modifiers,
            "enable_bonfires": self.enable_bonfires,
            "bonfire_count": self.bonfire_count,
            "bonfire_spots": self.bonfire_spots,
            "allows_mounts": self.allows_mounts,
            "requires_underwater_gear": self.requires_underwater_gear,
            "underwater_gear_id": self.underwater_gear_id,
            "enable_timed_events": self.enable_timed_events,
            "timed_event_duration": self.timed_event_duration,
            "bgm_loop_point": self.bgm_loop_point,
            "parallax_layers": self.parallax_layers,
            "enable_parallax": self.enable_parallax,
            "exploration_bonus": self.exploration_bonus,
            "item_respawn_time": self.item_respawn_time,
            "environment_performance_cost": self.environment_performance_cost,
            "supports_night_vision": self.supports_night_vision,
            "night_vision_filter": self.night_vision_filter,
            "god_rays_enabled": self.god_rays_enabled,
            "god_rays_intensity": self.god_rays_intensity,
            "destruction_physics": self.destruction_physics,
            "destruction_debris_duration": self.destruction_debris_duration,
            "scripts_on_clear": self.scripts_on_clear,
            "scripts_on_enter": self.scripts_on_enter,
            "scripts_on_exit": self.scripts_on_exit,
            "room_transformation_id": self.room_transformation_id,
            "dynamic_door_states": self.dynamic_door_states,
            "ally_spawn_points": self.ally_spawn_points,
            "ally_types": self.ally_types,
            "ally_chance": self.ally_chance,
            "power_required": self.power_required,
            "power_outlets": self.power_outlets,
            "mana_crystals_available": self.mana_crystals_available,
            "alternate_gravity_zones": self.alternate_gravity_zones,
            "minimum_wave_interval": self.minimum_wave_interval,
            "maximum_wave_interval": self.maximum_wave_interval,
            "enable_wave_scaling": self.enable_wave_scaling,
            "wave_scaling_factor": self.wave_scaling_factor,
            "wave_scaling_limit": self.wave_scaling_limit,
            "wave_infinite_mode": self.wave_infinite_mode,
            "cinematic_sequences": self.cinematic_sequences,
            "collectibles_count": self.collectibles_count,
            "collectibles_types": self.collectibles_types,
            "enforce_level_sync": self.enforce_level_sync,
            "level_sync_item": self.level_sync_item,
            "trigger_volume_count": self.trigger_volume_count,
            "trigger_volume_list": self.trigger_volume_list,
            "deactivate_on_completion": self.deactivate_on_completion,
            "completion_award_id": self.completion_award_id,
            "completion_skip_allowed": self.completion_skip_allowed,
            "linked_rooms": self.linked_rooms,
            "force_single_player": self.force_single_player,
            "has_side_quests": self.has_side_quests,
            "side_quest_ids": self.side_quest_ids,
            "can_build_structures": self.can_build_structures,
            "buildable_structure_ids": self.buildable_structure_ids,
            "enable_clan_system": self.enable_clan_system,
            "clan_ownership_required": self.clan_ownership_required,
            "default_spawn_region": self.default_spawn_region
        }

    def from_dict(self, data):
        self.room_id = data.get("room_id", self.room_id)
        self.width = data.get("width", self.width)
        self.height = data.get("height", self.height)
        self.theme = data.get("theme", self.theme)
        self.floor_texture = data.get("floor_texture", self.floor_texture)
        self.wall_texture = data.get("wall_texture", self.wall_texture)
        self.enemy_count = data.get("enemy_count", self.enemy_count)
        self.enemy_types = data.get("enemy_types", self.enemy_types)
        self.item_count = data.get("item_count", self.item_count)
        self.item_types = data.get("item_types", self.item_types)
        self.has_boss = data.get("has_boss", self.has_boss)
        self.boss_id = data.get("boss_id", self.boss_id)
        self.spawn_chance = data.get("spawn_chance", self.spawn_chance)
        self.min_player_level = data.get("min_player_level", self.min_player_level)
        self.environment_effects = data.get("environment_effects", self.environment_effects)
        self.lighting_type = data.get("lighting_type", self.lighting_type)
        self.is_safe_zone = data.get("is_safe_zone", self.is_safe_zone)
        self.scripted_event_id = data.get("scripted_event_id", self.scripted_event_id)
        self.event_trigger_chance = data.get("event_trigger_chance", self.event_trigger_chance)
        self.secret_rooms = data.get("secret_rooms", self.secret_rooms)
        self.secret_room_chance = data.get("secret_room_chance", self.secret_room_chance)
        self.secret_room_types = data.get("secret_room_types", self.secret_room_types)
        self.loot_multiplier = data.get("loot_multiplier", self.loot_multiplier)
        self.difficulty_scale = data.get("difficulty_scale", self.difficulty_scale)
        self.allow_resurgence = data.get("allow_resurgence", self.allow_resurgence)
        self.resurgence_cooldown = data.get("resurgence_cooldown", self.resurgence_cooldown)
        self.ambient_sound = data.get("ambient_sound", self.ambient_sound)
        self.background_music = data.get("background_music", self.background_music)
        self.background_music_volume = data.get("background_music_volume", self.background_music_volume)
        self.disable_minimap = data.get("disable_minimap", self.disable_minimap)
        self.override_minimap_color = data.get("override_minimap_color", self.override_minimap_color)
        self.override_minimap_icon = data.get("override_minimap_icon", self.override_minimap_icon)
        self.enable_puzzles = data.get("enable_puzzles", self.enable_puzzles)
        self.puzzle_count = data.get("puzzle_count", self.puzzle_count)
        self.puzzle_types = data.get("puzzle_types", self.puzzle_types)
        self.trap_count = data.get("trap_count", self.trap_count)
        self.trap_types = data.get("trap_types", self.trap_types)
        self.npc_count = data.get("npc_count", self.npc_count)
        self.npc_behaviors = data.get("npc_behaviors", self.npc_behaviors)
        self.room_tag = data.get("room_tag", self.room_tag)
        self.max_enemies_per_wave = data.get("max_enemies_per_wave", self.max_enemies_per_wave)
        self.wave_spawn_interval = data.get("wave_spawn_interval", self.wave_spawn_interval)
        self.allow_random_boss_spawn = data.get("allow_random_boss_spawn", self.allow_random_boss_spawn)
        self.random_boss_chance = data.get("random_boss_chance", self.random_boss_chance)
        self.fog_density = data.get("fog_density", self.fog_density)
        self.fog_color = data.get("fog_color", self.fog_color)
        self.gravity_override = data.get("gravity_override", self.gravity_override)
        self.physics_material = data.get("physics_material", self.physics_material)
        self.allow_climbing = data.get("allow_climbing", self.allow_climbing)
        self.locked = data.get("locked", self.locked)
        self.lock_key_id = data.get("lock_key_id", self.lock_key_id)
        self.auto_unlock_on_clear = data.get("auto_unlock_on_clear", self.auto_unlock_on_clear)
        self.reward_chest_count = data.get("reward_chest_count", self.reward_chest_count)
        self.reward_chest_types = data.get("reward_chest_types", self.reward_chest_types)
        self.environment_type = data.get("environment_type", self.environment_type)
        self.decoration_count = data.get("decoration_count", self.decoration_count)
        self.decoration_prefabs = data.get("decoration_prefabs", self.decoration_prefabs)
        self.prevent_fast_travel = data.get("prevent_fast_travel", self.prevent_fast_travel)
        self.ambient_light_color = data.get("ambient_light_color", self.ambient_light_color)
        self.ambient_light_intensity = data.get("ambient_light_intensity", self.ambient_light_intensity)
        self.post_process_profile = data.get("post_process_profile", self.post_process_profile)
        self.enemy_speed_multiplier = data.get("enemy_speed_multiplier", self.enemy_speed_multiplier)
        self.enemy_health_multiplier = data.get("enemy_health_multiplier", self.enemy_health_multiplier)
        self.enemy_attack_multiplier = data.get("enemy_attack_multiplier", self.enemy_attack_multiplier)
        self.allow_companion_spawn = data.get("allow_companion_spawn", self.allow_companion_spawn)
        self.companion_spawn_chance = data.get("companion_spawn_chance", self.companion_spawn_chance)
        self.room_connectors = data.get("room_connectors", self.room_connectors)
        self.room_script_path = data.get("room_script_path", self.room_script_path)
        self.terrain_variation = data.get("terrain_variation", self.terrain_variation)
        self.spawn_points = data.get("spawn_points", self.spawn_points)
        self.item_spots = data.get("item_spots", self.item_spots)
        self.trap_spots = data.get("trap_spots", self.trap_spots)
        self.decoration_spots = data.get("decoration_spots", self.decoration_spots)
        self.wave_count = data.get("wave_count", self.wave_count)
        self.wave_enemy_list = data.get("wave_enemy_list", self.wave_enemy_list)
        self.wave_boss_list = data.get("wave_boss_list", self.wave_boss_list)
        self.special_objectives = data.get("special_objectives", self.special_objectives)
        self.loot_drop_rate = data.get("loot_drop_rate", self.loot_drop_rate)
        self.enable_random_events = data.get("enable_random_events", self.enable_random_events)
        self.random_event_chance = data.get("random_event_chance", self.random_event_chance)
        self.random_event_types = data.get("random_event_types", self.random_event_types)
        self.theme_music_cue = data.get("theme_music_cue", self.theme_music_cue)
        self.ai_alert_radius_multiplier = data.get("ai_alert_radius_multiplier", self.ai_alert_radius_multiplier)
        self.ambient_particle_effect = data.get("ambient_particle_effect", self.ambient_particle_effect)
        self.level_requirement_override = data.get("level_requirement_override", self.level_requirement_override)
        self.custom_room_logic = data.get("custom_room_logic", self.custom_room_logic)
        self.custom_logic_path = data.get("custom_logic_path", self.custom_logic_path)
        self.max_players_allowed = data.get("max_players_allowed", self.max_players_allowed)
        self.enable_coop_triggers = data.get("enable_coop_triggers", self.enable_coop_triggers)
        self.spawner_limit = data.get("spawner_limit", self.spawner_limit)
        self.generation_weight = data.get("generation_weight", self.generation_weight)
        self.enable_portals = data.get("enable_portals", self.enable_portals)
        self.portal_count = data.get("portal_count", self.portal_count)
        self.portal_destinations = data.get("portal_destinations", self.portal_destinations)
        self.room_visibility_range = data.get("room_visibility_range", self.room_visibility_range)
        self.lockdown_on_entry = data.get("lockdown_on_entry", self.lockdown_on_entry)
        self.lockdown_duration = data.get("lockdown_duration", self.lockdown_duration)
        self.lockdown_release_condition = data.get("lockdown_release_condition", self.lockdown_release_condition)
        self.allow_air_dashes = data.get("allow_air_dashes", self.allow_air_dashes)
        self.allow_wall_jumps = data.get("allow_wall_jumps", self.allow_wall_jumps)
        self.surface_friction = data.get("surface_friction", self.surface_friction)
        self.bgm_fade_in_time = data.get("bgm_fade_in_time", self.bgm_fade_in_time)
        self.bgm_fade_out_time = data.get("bgm_fade_out_time", self.bgm_fade_out_time)
        self.has_water_zone = data.get("has_water_zone", self.has_water_zone)
        self.water_level = data.get("water_level", self.water_level)
        self.disable_player_abilities = data.get("disable_player_abilities", self.disable_player_abilities)
        self.allow_stealth_mechanics = data.get("allow_stealth_mechanics", self.allow_stealth_mechanics)
        self.interactable_objects = data.get("interactable_objects", self.interactable_objects)
        self.randomize_lights = data.get("randomize_lights", self.randomize_lights)
        self.light_color_options = data.get("light_color_options", self.light_color_options)
        self.light_intensity_min = data.get("light_intensity_min", self.light_intensity_min)
        self.light_intensity_max = data.get("light_intensity_max", self.light_intensity_max)
        self.loot_table_id = data.get("loot_table_id", self.loot_table_id)
        self.texture_variants = data.get("texture_variants", self.texture_variants)
        self.collision_map_path = data.get("collision_map_path", self.collision_map_path)
        self.enable_vertical_levels = data.get("enable_vertical_levels", self.enable_vertical_levels)
        self.vertical_section_count = data.get("vertical_section_count", self.vertical_section_count)
        self.randomize_enemies = data.get("randomize_enemies", self.randomize_enemies)
        self.random_enemy_pool = data.get("random_enemy_pool", self.random_enemy_pool)
        self.random_enemy_chance = data.get("random_enemy_chance", self.random_enemy_chance)
        self.post_room_script = data.get("post_room_script", self.post_room_script)
        self.ai_spawn_delay = data.get("ai_spawn_delay", self.ai_spawn_delay)
        self.death_zone_height = data.get("death_zone_height", self.death_zone_height)
        self.powerup_spot_count = data.get("powerup_spot_count", self.powerup_spot_count)
        self.powerup_spot_chance = data.get("powerup_spot_chance", self.powerup_spot_chance)
        self.powerup_types = data.get("powerup_types", self.powerup_types)
        self.random_room_attachments = data.get("random_room_attachments", self.random_room_attachments)
        self.custom_room_data = data.get("custom_room_data", self.custom_room_data)
        self.room_seed = data.get("room_seed", self.room_seed)
        self.room_seed_overrides_random = data.get("room_seed_overrides_random", self.room_seed_overrides_random)
        self.minimap_display_name = data.get("minimap_display_name", self.minimap_display_name)
        self.minimap_color_override = data.get("minimap_color_override", self.minimap_color_override)
        self.minimap_icon_override = data.get("minimap_icon_override", self.minimap_icon_override)
        self.access_requirements = data.get("access_requirements", self.access_requirements)
        self.procedural_lock_chance = data.get("procedural_lock_chance", self.procedural_lock_chance)
        self.lock_destruction_allowed = data.get("lock_destruction_allowed", self.lock_destruction_allowed)
        self.transition_cinematic_id = data.get("transition_cinematic_id", self.transition_cinematic_id)
        self.player_stat_modifiers = data.get("player_stat_modifiers", self.player_stat_modifiers)
        self.enable_bonfires = data.get("enable_bonfires", self.enable_bonfires)
        self.bonfire_count = data.get("bonfire_count", self.bonfire_count)
        self.bonfire_spots = data.get("bonfire_spots", self.bonfire_spots)
        self.allows_mounts = data.get("allows_mounts", self.allows_mounts)
        self.requires_underwater_gear = data.get("requires_underwater_gear", self.requires_underwater_gear)
        self.underwater_gear_id = data.get("underwater_gear_id", self.underwater_gear_id)
        self.enable_timed_events = data.get("enable_timed_events", self.enable_timed_events)
        self.timed_event_duration = data.get("timed_event_duration", self.timed_event_duration)
        self.bgm_loop_point = data.get("bgm_loop_point", self.bgm_loop_point)
        self.parallax_layers = data.get("parallax_layers", self.parallax_layers)
        self.enable_parallax = data.get("enable_parallax", self.enable_parallax)
        self.exploration_bonus = data.get("exploration_bonus", self.exploration_bonus)
        self.item_respawn_time = data.get("item_respawn_time", self.item_respawn_time)
        self.environment_performance_cost = data.get("environment_performance_cost", self.environment_performance_cost)
        self.supports_night_vision = data.get("supports_night_vision", self.supports_night_vision)
        self.night_vision_filter = data.get("night_vision_filter", self.night_vision_filter)
        self.god_rays_enabled = data.get("god_rays_enabled", self.god_rays_enabled)
        self.god_rays_intensity = data.get("god_rays_intensity", self.god_rays_intensity)
        self.destruction_physics = data.get("destruction_physics", self.destruction_physics)
        self.destruction_debris_duration = data.get("destruction_debris_duration", self.destruction_debris_duration)
        self.scripts_on_clear = data.get("scripts_on_clear", self.scripts_on_clear)
        self.scripts_on_enter = data.get("scripts_on_enter", self.scripts_on_enter)
        self.scripts_on_exit = data.get("scripts_on_exit", self.scripts_on_exit)
        self.room_transformation_id = data.get("room_transformation_id", self.room_transformation_id)
        self.dynamic_door_states = data.get("dynamic_door_states", self.dynamic_door_states)
        self.ally_spawn_points = data.get("ally_spawn_points", self.ally_spawn_points)
        self.ally_types = data.get("ally_types", self.ally_types)
        self.ally_chance = data.get("ally_chance", self.ally_chance)
        self.power_required = data.get("power_required", self.power_required)
        self.power_outlets = data.get("power_outlets", self.power_outlets)
        self.mana_crystals_available = data.get("mana_crystals_available", self.mana_crystals_available)
        self.alternate_gravity_zones = data.get("alternate_gravity_zones", self.alternate_gravity_zones)
        self.minimum_wave_interval = data.get("minimum_wave_interval", self.minimum_wave_interval)
        self.maximum_wave_interval = data.get("maximum_wave_interval", self.maximum_wave_interval)
        self.enable_wave_scaling = data.get("enable_wave_scaling", self.enable_wave_scaling)
        self.wave_scaling_factor = data.get("wave_scaling_factor", self.wave_scaling_factor)
        self.wave_scaling_limit = data.get("wave_scaling_limit", self.wave_scaling_limit)
        self.wave_infinite_mode = data.get("wave_infinite_mode", self.wave_infinite_mode)
        self.cinematic_sequences = data.get("cinematic_sequences", self.cinematic_sequences)
        self.collectibles_count = data.get("collectibles_count", self.collectibles_count)
        self.collectibles_types = data.get("collectibles_types", self.collectibles_types)
        self.enforce_level_sync = data.get("enforce_level_sync", self.enforce_level_sync)
        self.level_sync_item = data.get("level_sync_item", self.level_sync_item)
        self.trigger_volume_count = data.get("trigger_volume_count", self.trigger_volume_count)
        self.trigger_volume_list = data.get("trigger_volume_list", self.trigger_volume_list)
        self.deactivate_on_completion = data.get("deactivate_on_completion", self.deactivate_on_completion)
        self.completion_award_id = data.get("completion_award_id", self.completion_award_id)
        self.completion_skip_allowed = data.get("completion_skip_allowed", self.completion_skip_allowed)
        self.linked_rooms = data.get("linked_rooms", self.linked_rooms)
        self.force_single_player = data.get("force_single_player", self.force_single_player)
        self.has_side_quests = data.get("has_side_quests", self.has_side_quests)
        self.side_quest_ids = data.get("side_quest_ids", self.side_quest_ids)
        self.can_build_structures = data.get("can_build_structures", self.can_build_structures)
        self.buildable_structure_ids = data.get("buildable_structure_ids", self.buildable_structure_ids)
        self.enable_clan_system = data.get("enable_clan_system", self.enable_clan_system)
        self.clan_ownership_required = data.get("clan_ownership_required", self.clan_ownership_required)
        self.default_spawn_region = data.get("default_spawn_region", self.default_spawn_region)

class LevelConfig:
    def __init__(self):
        self.level_id = 1
        self.level_name = "Level1"
        self.max_rooms = 5
        self.room_configs = []
        self.level_theme = "dungeon"
        self.level_music = "level1_bgm.mp3"
        self.boss_music = "boss_theme.mp3"
        self.boss_room_id = None
        self.boss_data = {}
        self.required_items = []
        self.forbidden_items = []
        self.background_image = "bg_level1.png"
        self.minimap_icon = None
        self.transition_scene = None
        self.initial_player_position = (0, 0)
        self.gravity_scale = 1.0
        self.default_floor_texture = "floor.png"
        self.default_wall_texture = "wall.png"
        self.ambient_temperature = 20.0
        self.wind_speed = 0
        self.wind_direction = 0
        self.level_lighting_profile = "default"
        self.level_effects = []
        self.level_seed = None
        self.use_random_seed = True
        self.procedural_generation_method = "simple"
        self.max_enemies = 30
        self.enemy_wave_config = {}
        self.loot_table_override = None
        self.environment_damage_enabled = False
        self.environment_damage_amount = 0
        self.level_goal_description = ""
        self.on_level_start_scripts = []
        self.on_level_end_scripts = []
        self.time_limit_seconds = 0
        self.enable_time_limit = False
        self.exit_condition = "boss_defeated"
        self.multiplayer_allowed = True
        self.multiplayer_min_players = 1
        self.multiplayer_max_players = 4
        self.custom_data = {}
        self.enable_cutscenes = False
        self.cutscene_config = {}
        self.locked_until_story_progress = None
        self.story_flag_required = None
        self.unlock_story_flag = None
        self.faction_relations = {}
        self.spawn_logic_path = None
        self.dynamic_lights_enabled = True
        self.ai_difficulty_modifier = 1.0
        self.boss_health_scale = 1.0
        self.boss_damage_scale = 1.0
        self.boss_reward_items = []
        self.boss_drops_loot = True
        self.randomize_boss_attributes = False
        self.parallax_background = []
        self.survival_mode = False
        self.survival_waves = 0
        self.survival_interval = 0
        self.enable_weather_randomization = False
        self.level_achievements = []
        self.mandatory_rooms = []
        self.allow_room_revisit = True
        self.room_revisit_cooldown = 0
        self.item_spawn_chance = 1.0
        self.ground_type = "stone"
        self.hazard_spots = []
        self.enable_secret_exits = False
        self.secret_exit_count = 0
        self.secret_exit_rooms = []
        self.override_spawn_logic = False
        self.room_order_fixed = False
        self.fog_settings = {}
        self.override_camera_behavior = False
        self.camera_behavior_config = {}
        self.enable_intro_sequence = False
        self.intro_sequence_id = None
        self.enable_outro_sequence = False
        self.outro_sequence_id = None
        self.restrict_consumables = False
        self.restricted_consumables_list = []
        self.preferred_companion_id = None
        self.level_clear_script = None
        self.allow_map_access = True
        self.map_display_name = ""
        self.map_icon_path = None
        self.enable_story_dialogues = True

    def to_dict(self):
        return {
            "level_id": self.level_id,
            "level_name": self.level_name,
            "max_rooms": self.max_rooms,
            "room_configs": [r.to_dict() for r in self.room_configs],
            "level_theme": self.level_theme,
            "level_music": self.level_music,
            "boss_music": self.boss_music,
            "boss_room_id": self.boss_room_id,
            "boss_data": self.boss_data,
            "required_items": self.required_items,
            "forbidden_items": self.forbidden_items,
            "background_image": self.background_image,
            "minimap_icon": self.minimap_icon,
            "transition_scene": self.transition_scene,
            "initial_player_position": self.initial_player_position,
            "gravity_scale": self.gravity_scale,
            "default_floor_texture": self.default_floor_texture,
            "default_wall_texture": self.default_wall_texture,
            "ambient_temperature": self.ambient_temperature,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "level_lighting_profile": self.level_lighting_profile,
            "level_effects": self.level_effects,
            "level_seed": self.level_seed,
            "use_random_seed": self.use_random_seed,
            "procedural_generation_method": self.procedural_generation_method,
            "max_enemies": self.max_enemies,
            "enemy_wave_config": self.enemy_wave_config,
            "loot_table_override": self.loot_table_override,
            "environment_damage_enabled": self.environment_damage_enabled,
            "environment_damage_amount": self.environment_damage_amount,
            "level_goal_description": self.level_goal_description,
            "on_level_start_scripts": self.on_level_start_scripts,
            "on_level_end_scripts": self.on_level_end_scripts,
            "time_limit_seconds": self.time_limit_seconds,
            "enable_time_limit": self.enable_time_limit,
            "exit_condition": self.exit_condition,
            "multiplayer_allowed": self.multiplayer_allowed,
            "multiplayer_min_players": self.multiplayer_min_players,
            "multiplayer_max_players": self.multiplayer_max_players,
            "custom_data": self.custom_data,
            "enable_cutscenes": self.enable_cutscenes,
            "cutscene_config": self.cutscene_config,
            "locked_until_story_progress": self.locked_until_story_progress,
            "story_flag_required": self.story_flag_required,
            "unlock_story_flag": self.unlock_story_flag,
            "faction_relations": self.faction_relations,
            "spawn_logic_path": self.spawn_logic_path,
            "dynamic_lights_enabled": self.dynamic_lights_enabled,
            "ai_difficulty_modifier": self.ai_difficulty_modifier,
            "boss_health_scale": self.boss_health_scale,
            "boss_damage_scale": self.boss_damage_scale,
            "boss_reward_items": self.boss_reward_items,
            "boss_drops_loot": self.boss_drops_loot,
            "randomize_boss_attributes": self.randomize_boss_attributes,
            "parallax_background": self.parallax_background,
            "survival_mode": self.survival_mode,
            "survival_waves": self.survival_waves,
            "survival_interval": self.survival_interval,
            "enable_weather_randomization": self.enable_weather_randomization,
            "level_achievements": self.level_achievements,
            "mandatory_rooms": self.mandatory_rooms,
            "allow_room_revisit": self.allow_room_revisit,
            "room_revisit_cooldown": self.room_revisit_cooldown,
            "item_spawn_chance": self.item_spawn_chance,
            "ground_type": self.ground_type,
            "hazard_spots": self.hazard_spots,
            "enable_secret_exits": self.enable_secret_exits,
            "secret_exit_count": self.secret_exit_count,
            "secret_exit_rooms": self.secret_exit_rooms,
            "override_spawn_logic": self.override_spawn_logic,
            "room_order_fixed": self.room_order_fixed,
            "fog_settings": self.fog_settings,
            "override_camera_behavior": self.override_camera_behavior,
            "camera_behavior_config": self.camera_behavior_config,
            "enable_intro_sequence": self.enable_intro_sequence,
            "intro_sequence_id": self.intro_sequence_id,
            "enable_outro_sequence": self.enable_outro_sequence,
            "outro_sequence_id": self.outro_sequence_id,
            "restrict_consumables": self.restrict_consumables,
            "restricted_consumables_list": self.restricted_consumables_list,
            "preferred_companion_id": self.preferred_companion_id,
            "level_clear_script": self.level_clear_script,
            "allow_map_access": self.allow_map_access,
            "map_display_name": self.map_display_name,
            "map_icon_path": self.map_icon_path,
            "enable_story_dialogues": self.enable_story_dialogues
        }

    def from_dict(self, data):
        self.level_id = data.get("level_id", self.level_id)
        self.level_name = data.get("level_name", self.level_name)
        self.max_rooms = data.get("max_rooms", self.max_rooms)
        if "room_configs" in data:
            self.room_configs = []
            for rc in data["room_configs"]:
                c = RoomConfig()
                c.from_dict(rc)
                self.room_configs.append(c)
        self.level_theme = data.get("level_theme", self.level_theme)
        self.level_music = data.get("level_music", self.level_music)
        self.boss_music = data.get("boss_music", self.boss_music)
        self.boss_room_id = data.get("boss_room_id", self.boss_room_id)
        self.boss_data = data.get("boss_data", self.boss_data)
        self.required_items = data.get("required_items", self.required_items)
        self.forbidden_items = data.get("forbidden_items", self.forbidden_items)
        self.background_image = data.get("background_image", self.background_image)
        self.minimap_icon = data.get("minimap_icon", self.minimap_icon)
        self.transition_scene = data.get("transition_scene", self.transition_scene)
        self.initial_player_position = data.get("initial_player_position", self.initial_player_position)
        self.gravity_scale = data.get("gravity_scale", self.gravity_scale)
        self.default_floor_texture = data.get("default_floor_texture", self.default_floor_texture)
        self.default_wall_texture = data.get("default_wall_texture", self.default_wall_texture)
        self.ambient_temperature = data.get("ambient_temperature", self.ambient_temperature)
        self.wind_speed = data.get("wind_speed", self.wind_speed)
        self.wind_direction = data.get("wind_direction", self.wind_direction)
        self.level_lighting_profile = data.get("level_lighting_profile", self.level_lighting_profile)
        self.level_effects = data.get("level_effects", self.level_effects)
        self.level_seed = data.get("level_seed", self.level_seed)
        self.use_random_seed = data.get("use_random_seed", self.use_random_seed)
        self.procedural_generation_method = data.get("procedural_generation_method", self.procedural_generation_method)
        self.max_enemies = data.get("max_enemies", self.max_enemies)
        self.enemy_wave_config = data.get("enemy_wave_config", self.enemy_wave_config)
        self.loot_table_override = data.get("loot_table_override", self.loot_table_override)
        self.environment_damage_enabled = data.get("environment_damage_enabled", self.environment_damage_enabled)
        self.environment_damage_amount = data.get("environment_damage_amount", self.environment_damage_amount)
        self.level_goal_description = data.get("level_goal_description", self.level_goal_description)
        self.on_level_start_scripts = data.get("on_level_start_scripts", self.on_level_start_scripts)
        self.on_level_end_scripts = data.get("on_level_end_scripts", self.on_level_end_scripts)
        self.time_limit_seconds = data.get("time_limit_seconds", self.time_limit_seconds)
        self.enable_time_limit = data.get("enable_time_limit", self.enable_time_limit)
        self.exit_condition = data.get("exit_condition", self.exit_condition)
        self.multiplayer_allowed = data.get("multiplayer_allowed", self.multiplayer_allowed)
        self.multiplayer_min_players = data.get("multiplayer_min_players", self.multiplayer_min_players)
        self.multiplayer_max_players = data.get("multiplayer_max_players", self.multiplayer_max_players)
        self.custom_data = data.get("custom_data", self.custom_data)
        self.enable_cutscenes = data.get("enable_cutscenes", self.enable_cutscenes)
        self.cutscene_config = data.get("cutscene_config", self.cutscene_config)
        self.locked_until_story_progress = data.get("locked_until_story_progress", self.locked_until_story_progress)
        self.story_flag_required = data.get("story_flag_required", self.story_flag_required)
        self.unlock_story_flag = data.get("unlock_story_flag", self.unlock_story_flag)
        self.faction_relations = data.get("faction_relations", self.faction_relations)
        self.spawn_logic_path = data.get("spawn_logic_path", self.spawn_logic_path)
        self.dynamic_lights_enabled = data.get("dynamic_lights_enabled", self.dynamic_lights_enabled)
        self.ai_difficulty_modifier = data.get("ai_difficulty_modifier", self.ai_difficulty_modifier)
        self.boss_health_scale = data.get("boss_health_scale", self.boss_health_scale)
        self.boss_damage_scale = data.get("boss_damage_scale", self.boss_damage_scale)
        self.boss_reward_items = data.get("boss_reward_items", self.boss_reward_items)
        self.boss_drops_loot = data.get("boss_drops_loot", self.boss_drops_loot)
        self.randomize_boss_attributes = data.get("randomize_boss_attributes", self.randomize_boss_attributes)
        self.parallax_background = data.get("parallax_background", self.parallax_background)
        self.survival_mode = data.get("survival_mode", self.survival_mode)
        self.survival_waves = data.get("survival_waves", self.survival_waves)
        self.survival_interval = data.get("survival_interval", self.survival_interval)
        self.enable_weather_randomization = data.get("enable_weather_randomization", self.enable_weather_randomization)
        self.level_achievements = data.get("level_achievements", self.level_achievements)
        self.mandatory_rooms = data.get("mandatory_rooms", self.mandatory_rooms)
        self.allow_room_revisit = data.get("allow_room_revisit", self.allow_room_revisit)
        self.room_revisit_cooldown = data.get("room_revisit_cooldown", self.room_revisit_cooldown)
        self.item_spawn_chance = data.get("item_spawn_chance", self.item_spawn_chance)
        self.ground_type = data.get("ground_type", self.ground_type)
        self.hazard_spots = data.get("hazard_spots", self.hazard_spots)
        self.enable_secret_exits = data.get("enable_secret_exits", self.enable_secret_exits)
        self.secret_exit_count = data.get("secret_exit_count", self.secret_exit_count)
        self.secret_exit_rooms = data.get("secret_exit_rooms", self.secret_exit_rooms)
        self.override_spawn_logic = data.get("override_spawn_logic", self.override_spawn_logic)
        self.room_order_fixed = data.get("room_order_fixed", self.room_order_fixed)
        self.fog_settings = data.get("fog_settings", self.fog_settings)
        self.override_camera_behavior = data.get("override_camera_behavior", self.override_camera_behavior)
        self.camera_behavior_config = data.get("camera_behavior_config", self.camera_behavior_config)
        self.enable_intro_sequence = data.get("enable_intro_sequence", self.enable_intro_sequence)
        self.intro_sequence_id = data.get("intro_sequence_id", self.intro_sequence_id)
        self.enable_outro_sequence = data.get("enable_outro_sequence", self.enable_outro_sequence)
        self.outro_sequence_id = data.get("outro_sequence_id", self.outro_sequence_id)
        self.restrict_consumables = data.get("restrict_consumables", self.restrict_consumables)
        self.restricted_consumables_list = data.get("restricted_consumables_list", self.restricted_consumables_list)
        self.preferred_companion_id = data.get("preferred_companion_id", self.preferred_companion_id)
        self.level_clear_script = data.get("level_clear_script", self.level_clear_script)
        self.allow_map_access = data.get("allow_map_access", self.allow_map_access)
        self.map_display_name = data.get("map_display_name", self.map_display_name)
        self.map_icon_path = data.get("map_icon_path", self.map_icon_path)
        self.enable_story_dialogues = data.get("enable_story_dialogues", self.enable_story_dialogues)

class LevelManager:
    def __init__(self):
        self.levels = []
        self.current_level_index = 0
        self.level_config_file = "level_config.json"
        self.backup_on_save = True
        self.backup_folder = "level_backups"
        self.last_load_time = None
        self.last_save_time = None
        self.enable_encryption = False
        self.encryption_key = None
        self.load_error_count = 0
        self.max_load_errors = 3
        self.enable_threaded_loading = False
        self.lock_file_path = "level_config.lock"
        self.enable_lock_file = False
        self.lock_file_handle = None

    def load_levels(self):
        if not os.path.exists(self.level_config_file):
            return
        try:
            with open(self.level_config_file, "r", encoding="utf-8") as f:
                if self.enable_encryption and self.encryption_key:
                    raw = f.read()
                    decrypted = self.decrypt(raw)
                    data = json.loads(decrypted)
                else:
                    data = json.load(f)
            self.levels = []
            for lvl_data in data.get("levels", []):
                lvl = LevelConfig()
                lvl.from_dict(lvl_data)
                self.levels.append(lvl)
            self.current_level_index = data.get("current_level_index", self.current_level_index)
            self.last_load_time = time.time()
        except:
            self.load_error_count += 1

    def save_levels(self):
        if self.load_error_count >= self.max_load_errors:
            return
        data = {
            "levels": [l.to_dict() for l in self.levels],
            "current_level_index": self.current_level_index
        }
        if self.backup_on_save and os.path.exists(self.level_config_file):
            if not os.path.exists(self.backup_folder):
                os.makedirs(self.backup_folder)
            backup_file = os.path.join(self.backup_folder, f"level_config_backup_{int(time.time())}.json")
            shutil.copy(self.level_config_file, backup_file)
        if self.enable_encryption and self.encryption_key:
            encoded = json.dumps(data)
            encrypted = self.encrypt(encoded)
            with open(self.level_config_file, "w", encoding="utf-8") as f:
                f.write(encrypted)
        else:
            with open(self.level_config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        self.last_save_time = time.time()

    def encrypt(self, data):
        if not self.encryption_key:
            return data
        r = []
        for i, c in enumerate(data):
            r.append(chr(ord(c) ^ ord(self.encryption_key[i % len(self.encryption_key)])))
        return "".join(r)

    def decrypt(self, data):
        if not self.encryption_key:
            return data
        r = []
        for i, c in enumerate(data):
            r.append(chr(ord(c) ^ ord(self.encryption_key[i % len(self.encryption_key)])))
        return "".join(r)

    def acquire_lock(self):
        if not self.enable_lock_file:
            return
        if os.path.exists(self.lock_file_path):
            return
        self.lock_file_handle = open(self.lock_file_path, "w")
        self.lock_file_handle.write(str(os.getpid()))
        self.lock_file_handle.flush()

    def release_lock(self):
        if not self.enable_lock_file:
            return
        if self.lock_file_handle:
            self.lock_file_handle.close()
        if os.path.exists(self.lock_file_path):
            os.remove(self.lock_file_path)

    def get_current_level(self):
        if 0 <= self.current_level_index < len(self.levels):
            return self.levels[self.current_level_index]
        return None

    def set_current_level_index(self, index):
        if 0 <= index < len(self.levels):
            self.current_level_index = index

    def next_level(self):
        if self.current_level_index < len(self.levels) - 1:
            self.current_level_index += 1

    def previous_level(self):
        if self.current_level_index > 0:
            self.current_level_index -= 1

    def create_new_level(self, level_id, level_name):
        lvl = LevelConfig()
        lvl.level_id = level_id
        lvl.level_name = level_name
        self.levels.append(lvl)

    def remove_level(self, level_id):
        self.levels = [lvl for lvl in self.levels if lvl.level_id != level_id]

    def find_level_by_id(self, level_id):
        for lvl in self.levels:
            if lvl.level_id == level_id:
                return lvl
        return None

    def refresh_level_seeds(self):
        for lvl in self.levels:
            if lvl.use_random_seed:
                lvl.level_seed = random.randint(0, 999999999)
            for rc in lvl.room_configs:
                if rc.room_seed_overrides_random is False:
                    rc.room_seed = random.randint(0, 999999999)

    def randomize_level_order(self):
        random.shuffle(self.levels)
        self.current_level_index = 0

    def verify_levels(self):
        pass

    def initialize(self):
        self.acquire_lock()
        self.load_levels()
        self.verify_levels()

    def finalize(self):
        self.release_lock()
