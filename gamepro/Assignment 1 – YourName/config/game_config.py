import os
import sys
import json
import random
import platform
import uuid
import time
import shutil
import pathlib
import subprocess
import math
import threading
import functools
import tempfile
import getpass
import secrets

class WindowConfig:
    def __init__(self):
        self.fullscreen = False
        self.width = 1280
        self.height = 720
        self.resizable = True
        self.title = "GameWindow"
        self.icon_path = ""
        self.frame_rate = 60
        self.vsync = True
        self.use_custom_cursor = False
        self.cursor_path = ""
        self.window_position_x = None
        self.window_position_y = None
        self.borderless = False
        self.monitor_index = 0
        self.maximized = False
        self.minimized = False
        self.high_dpi = False
        self.force_aspect_ratio = False
        self.aspect_width = 16
        self.aspect_height = 9
        self.auto_scale_ui = False
        self.scale_ui_factor = 1.0
        self.max_fps_unfocused = 30
        self.hide_cursor_when_inactive = False
        self.ignore_display_scaling = False
        self.gpu_index = 0
        self.support_hdr = False
        self.window_opacity = 1.0
        self.allow_alt_enter_toggle = True
        self.enable_quick_resize_hotkeys = False
        self.hotkey_resize_step = 100
        self.enable_custom_gamma = False
        self.custom_gamma_value = 1.0
        self.allow_drag_resize = True
        self.constrain_cursor_to_window = False
        self.enable_window_snapping = False
        self.snap_threshold = 20
        self.restore_last_window_position = False
        self.last_position_file = "window_pos.json"

    def to_dict(self):
        return {
            "fullscreen": self.fullscreen,
            "width": self.width,
            "height": self.height,
            "resizable": self.resizable,
            "title": self.title,
            "icon_path": self.icon_path,
            "frame_rate": self.frame_rate,
            "vsync": self.vsync,
            "use_custom_cursor": self.use_custom_cursor,
            "cursor_path": self.cursor_path,
            "window_position_x": self.window_position_x,
            "window_position_y": self.window_position_y,
            "borderless": self.borderless,
            "monitor_index": self.monitor_index,
            "maximized": self.maximized,
            "minimized": self.minimized,
            "high_dpi": self.high_dpi,
            "force_aspect_ratio": self.force_aspect_ratio,
            "aspect_width": self.aspect_width,
            "aspect_height": self.aspect_height,
            "auto_scale_ui": self.auto_scale_ui,
            "scale_ui_factor": self.scale_ui_factor,
            "max_fps_unfocused": self.max_fps_unfocused,
            "hide_cursor_when_inactive": self.hide_cursor_when_inactive,
            "ignore_display_scaling": self.ignore_display_scaling,
            "gpu_index": self.gpu_index,
            "support_hdr": self.support_hdr,
            "window_opacity": self.window_opacity,
            "allow_alt_enter_toggle": self.allow_alt_enter_toggle,
            "enable_quick_resize_hotkeys": self.enable_quick_resize_hotkeys,
            "hotkey_resize_step": self.hotkey_resize_step,
            "enable_custom_gamma": self.enable_custom_gamma,
            "custom_gamma_value": self.custom_gamma_value,
            "allow_drag_resize": self.allow_drag_resize,
            "constrain_cursor_to_window": self.constrain_cursor_to_window,
            "enable_window_snapping": self.enable_window_snapping,
            "snap_threshold": self.snap_threshold,
            "restore_last_window_position": self.restore_last_window_position,
            "last_position_file": self.last_position_file
        }

    def from_dict(self, data):
        self.fullscreen = data.get("fullscreen", self.fullscreen)
        self.width = data.get("width", self.width)
        self.height = data.get("height", self.height)
        self.resizable = data.get("resizable", self.resizable)
        self.title = data.get("title", self.title)
        self.icon_path = data.get("icon_path", self.icon_path)
        self.frame_rate = data.get("frame_rate", self.frame_rate)
        self.vsync = data.get("vsync", self.vsync)
        self.use_custom_cursor = data.get("use_custom_cursor", self.use_custom_cursor)
        self.cursor_path = data.get("cursor_path", self.cursor_path)
        self.window_position_x = data.get("window_position_x", self.window_position_x)
        self.window_position_y = data.get("window_position_y", self.window_position_y)
        self.borderless = data.get("borderless", self.borderless)
        self.monitor_index = data.get("monitor_index", self.monitor_index)
        self.maximized = data.get("maximized", self.maximized)
        self.minimized = data.get("minimized", self.minimized)
        self.high_dpi = data.get("high_dpi", self.high_dpi)
        self.force_aspect_ratio = data.get("force_aspect_ratio", self.force_aspect_ratio)
        self.aspect_width = data.get("aspect_width", self.aspect_width)
        self.aspect_height = data.get("aspect_height", self.aspect_height)
        self.auto_scale_ui = data.get("auto_scale_ui", self.auto_scale_ui)
        self.scale_ui_factor = data.get("scale_ui_factor", self.scale_ui_factor)
        self.max_fps_unfocused = data.get("max_fps_unfocused", self.max_fps_unfocused)
        self.hide_cursor_when_inactive = data.get("hide_cursor_when_inactive", self.hide_cursor_when_inactive)
        self.ignore_display_scaling = data.get("ignore_display_scaling", self.ignore_display_scaling)
        self.gpu_index = data.get("gpu_index", self.gpu_index)
        self.support_hdr = data.get("support_hdr", self.support_hdr)
        self.window_opacity = data.get("window_opacity", self.window_opacity)
        self.allow_alt_enter_toggle = data.get("allow_alt_enter_toggle", self.allow_alt_enter_toggle)
        self.enable_quick_resize_hotkeys = data.get("enable_quick_resize_hotkeys", self.enable_quick_resize_hotkeys)
        self.hotkey_resize_step = data.get("hotkey_resize_step", self.hotkey_resize_step)
        self.enable_custom_gamma = data.get("enable_custom_gamma", self.enable_custom_gamma)
        self.custom_gamma_value = data.get("custom_gamma_value", self.custom_gamma_value)
        self.allow_drag_resize = data.get("allow_drag_resize", self.allow_drag_resize)
        self.constrain_cursor_to_window = data.get("constrain_cursor_to_window", self.constrain_cursor_to_window)
        self.enable_window_snapping = data.get("enable_window_snapping", self.enable_window_snapping)
        self.snap_threshold = data.get("snap_threshold", self.snap_threshold)
        self.restore_last_window_position = data.get("restore_last_window_position", self.restore_last_window_position)
        self.last_position_file = data.get("last_position_file", self.last_position_file)

class AudioConfig:
    def __init__(self):
        self.master_volume = 1.0
        self.music_volume = 0.8
        self.sfx_volume = 0.9
        self.ambient_volume = 0.7
        self.mute_all = False
        self.audio_device = None
        self.enable_spatial_audio = False
        self.spatial_audio_distance_scale = 1.0
        self.spatial_audio_doppler_scale = 1.0
        self.spatial_audio_rolloff_scale = 1.0
        self.music_enabled = True
        self.sfx_enabled = True
        self.ambient_enabled = True
        self.output_channels = 2
        self.use_hrtf = False
        self.limit_sfx_voices = False
        self.max_sfx_voices = 32
        self.fade_music_when_paused = True
        self.crossfade_music = False
        self.crossfade_duration = 1.0
        self.high_quality_resampling = True
        self.music_format = "mp3"
        self.preload_audio_assets = False
        self.dynamic_range_compression = False
        self.compression_threshold = 0.8
        self.compression_ratio = 2.0
        self.loudness_normalization = False
        self.normalization_level = -14
        self.bass_boost = False
        self.bass_boost_amount = 0.5
        self.treble_boost = False
        self.treble_boost_amount = 0.5
        self.equalizer_enabled = False
        self.equalizer_bands = [0.0 for _ in range(10)]
        self.auto_detect_best_device = True
        self.output_latency = 0.05
        self.disable_audio_on_focus_loss = False
        self.audio_thread_priority = "normal"
        self.reload_audio_on_hotplug = True
        self.prefer_async_loading = False
        self.buffer_size_frames = 512
        self.sample_rate = 44100
        self.virtualize_when_minimized = True
        self.limit_3d_distance = 1000.0
        self.rolloff_model = "logarithmic"
        self.enable_audio_filters = False
        self.audio_filters = []
        self.hrtf_quality = "medium"
        self.enable_reverb = False
        self.reverb_preset = "smallroom"

    def to_dict(self):
        return {
            "master_volume": self.master_volume,
            "music_volume": self.music_volume,
            "sfx_volume": self.sfx_volume,
            "ambient_volume": self.ambient_volume,
            "mute_all": self.mute_all,
            "audio_device": self.audio_device,
            "enable_spatial_audio": self.enable_spatial_audio,
            "spatial_audio_distance_scale": self.spatial_audio_distance_scale,
            "spatial_audio_doppler_scale": self.spatial_audio_doppler_scale,
            "spatial_audio_rolloff_scale": self.spatial_audio_rolloff_scale,
            "music_enabled": self.music_enabled,
            "sfx_enabled": self.sfx_enabled,
            "ambient_enabled": self.ambient_enabled,
            "output_channels": self.output_channels,
            "use_hrtf": self.use_hrtf,
            "limit_sfx_voices": self.limit_sfx_voices,
            "max_sfx_voices": self.max_sfx_voices,
            "fade_music_when_paused": self.fade_music_when_paused,
            "crossfade_music": self.crossfade_music,
            "crossfade_duration": self.crossfade_duration,
            "high_quality_resampling": self.high_quality_resampling,
            "music_format": self.music_format,
            "preload_audio_assets": self.preload_audio_assets,
            "dynamic_range_compression": self.dynamic_range_compression,
            "compression_threshold": self.compression_threshold,
            "compression_ratio": self.compression_ratio,
            "loudness_normalization": self.loudness_normalization,
            "normalization_level": self.normalization_level,
            "bass_boost": self.bass_boost,
            "bass_boost_amount": self.bass_boost_amount,
            "treble_boost": self.treble_boost,
            "treble_boost_amount": self.treble_boost_amount,
            "equalizer_enabled": self.equalizer_enabled,
            "equalizer_bands": self.equalizer_bands,
            "auto_detect_best_device": self.auto_detect_best_device,
            "output_latency": self.output_latency,
            "disable_audio_on_focus_loss": self.disable_audio_on_focus_loss,
            "audio_thread_priority": self.audio_thread_priority,
            "reload_audio_on_hotplug": self.reload_audio_on_hotplug,
            "prefer_async_loading": self.prefer_async_loading,
            "buffer_size_frames": self.buffer_size_frames,
            "sample_rate": self.sample_rate,
            "virtualize_when_minimized": self.virtualize_when_minimized,
            "limit_3d_distance": self.limit_3d_distance,
            "rolloff_model": self.rolloff_model,
            "enable_audio_filters": self.enable_audio_filters,
            "audio_filters": self.audio_filters,
            "hrtf_quality": self.hrtf_quality,
            "enable_reverb": self.enable_reverb,
            "reverb_preset": self.reverb_preset
        }

    def from_dict(self, data):
        self.master_volume = data.get("master_volume", self.master_volume)
        self.music_volume = data.get("music_volume", self.music_volume)
        self.sfx_volume = data.get("sfx_volume", self.sfx_volume)
        self.ambient_volume = data.get("ambient_volume", self.ambient_volume)
        self.mute_all = data.get("mute_all", self.mute_all)
        self.audio_device = data.get("audio_device", self.audio_device)
        self.enable_spatial_audio = data.get("enable_spatial_audio", self.enable_spatial_audio)
        self.spatial_audio_distance_scale = data.get("spatial_audio_distance_scale", self.spatial_audio_distance_scale)
        self.spatial_audio_doppler_scale = data.get("spatial_audio_doppler_scale", self.spatial_audio_doppler_scale)
        self.spatial_audio_rolloff_scale = data.get("spatial_audio_rolloff_scale", self.spatial_audio_rolloff_scale)
        self.music_enabled = data.get("music_enabled", self.music_enabled)
        self.sfx_enabled = data.get("sfx_enabled", self.sfx_enabled)
        self.ambient_enabled = data.get("ambient_enabled", self.ambient_enabled)
        self.output_channels = data.get("output_channels", self.output_channels)
        self.use_hrtf = data.get("use_hrtf", self.use_hrtf)
        self.limit_sfx_voices = data.get("limit_sfx_voices", self.limit_sfx_voices)
        self.max_sfx_voices = data.get("max_sfx_voices", self.max_sfx_voices)
        self.fade_music_when_paused = data.get("fade_music_when_paused", self.fade_music_when_paused)
        self.crossfade_music = data.get("crossfade_music", self.crossfade_music)
        self.crossfade_duration = data.get("crossfade_duration", self.crossfade_duration)
        self.high_quality_resampling = data.get("high_quality_resampling", self.high_quality_resampling)
        self.music_format = data.get("music_format", self.music_format)
        self.preload_audio_assets = data.get("preload_audio_assets", self.preload_audio_assets)
        self.dynamic_range_compression = data.get("dynamic_range_compression", self.dynamic_range_compression)
        self.compression_threshold = data.get("compression_threshold", self.compression_threshold)
        self.compression_ratio = data.get("compression_ratio", self.compression_ratio)
        self.loudness_normalization = data.get("loudness_normalization", self.loudness_normalization)
        self.normalization_level = data.get("normalization_level", self.normalization_level)
        self.bass_boost = data.get("bass_boost", self.bass_boost)
        self.bass_boost_amount = data.get("bass_boost_amount", self.bass_boost_amount)
        self.treble_boost = data.get("treble_boost", self.treble_boost)
        self.treble_boost_amount = data.get("treble_boost_amount", self.treble_boost_amount)
        self.equalizer_enabled = data.get("equalizer_enabled", self.equalizer_enabled)
        self.equalizer_bands = data.get("equalizer_bands", self.equalizer_bands)
        self.auto_detect_best_device = data.get("auto_detect_best_device", self.auto_detect_best_device)
        self.output_latency = data.get("output_latency", self.output_latency)
        self.disable_audio_on_focus_loss = data.get("disable_audio_on_focus_loss", self.disable_audio_on_focus_loss)
        self.audio_thread_priority = data.get("audio_thread_priority", self.audio_thread_priority)
        self.reload_audio_on_hotplug = data.get("reload_audio_on_hotplug", self.reload_audio_on_hotplug)
        self.prefer_async_loading = data.get("prefer_async_loading", self.prefer_async_loading)
        self.buffer_size_frames = data.get("buffer_size_frames", self.buffer_size_frames)
        self.sample_rate = data.get("sample_rate", self.sample_rate)
        self.virtualize_when_minimized = data.get("virtualize_when_minimized", self.virtualize_when_minimized)
        self.limit_3d_distance = data.get("limit_3d_distance", self.limit_3d_distance)
        self.rolloff_model = data.get("rolloff_model", self.rolloff_model)
        self.enable_audio_filters = data.get("enable_audio_filters", self.enable_audio_filters)
        self.audio_filters = data.get("audio_filters", self.audio_filters)
        self.hrtf_quality = data.get("hrtf_quality", self.hrtf_quality)
        self.enable_reverb = data.get("enable_reverb", self.enable_reverb)
        self.reverb_preset = data.get("reverb_preset", self.reverb_preset)

class GameplayConfig:
    def __init__(self):
        self.difficulty = "normal"
        self.start_lives = 3
        self.max_level = 10
        self.enable_tutorial = True
        self.auto_save = True
        self.save_slot = 0
        self.language = "en"
        self.allow_cheats = False
        self.enable_spawn_randomization = True
        self.enable_weather_system = True
        self.auto_adjust_difficulty = False
        self.random_seed = None
        self.enable_nightmare_mode = False
        self.start_resources = 100
        self.resource_rate_multiplier = 1.0
        self.enemy_spawn_rate = 1.0
        self.enemy_damage_multiplier = 1.0
        self.player_damage_multiplier = 1.0
        self.enable_friendly_fire = False
        self.enable_auto_loot = False
        self.loot_multiplier = 1.0
        self.boss_health_multiplier = 1.0
        self.boss_damage_multiplier = 1.0
        self.unlock_all_levels = False
        self.unlock_all_weapons = False
        self.enable_rogue_like_mode = False
        self.unlock_all_buildings = False
        self.enable_building_upgrades = True
        self.ally_spawn_rate = 0.0
        self.enable_save_every_level = True
        self.enable_achievements = True
        self.enable_prestige_system = False
        self.prestige_multiplier = 1.0
        self.enable_leaderboards = False
        self.enable_speedrun_timers = False
        self.enable_multi_language_support = True
        self.enable_mod_support = False
        self.mod_folder_path = "mods"
        self.enable_auto_update = False
        self.enable_online_features = False
        self.enable_voice_chat = False
        self.enable_cross_platform_play = False
        self.enable_controller_vibration = True
        self.vibration_strength = 0.5
        self.max_active_quests = 3
        self.quest_reward_multiplier = 1.0
        self.enable_dynamic_lighting = True
        self.enable_realistic_physics = False
        self.physics_detail_level = 1
        self.enable_procedural_generation = True
        self.procedural_seed = 0
        self.enable_fog_of_war = False
        self.enable_auto_farming = False
        self.enable_customizable_difficulty = False
        self.custom_difficulty_parameters = {}
        self.enable_loot_boxes = False
        self.enable_energy_system = False
        self.energy_system_capacity = 100
        self.energy_recharge_rate = 1
        self.enable_weather_variations = False
        self.enable_npc_dialogues = True

    def to_dict(self):
        return {
            "difficulty": self.difficulty,
            "start_lives": self.start_lives,
            "max_level": self.max_level,
            "enable_tutorial": self.enable_tutorial,
            "auto_save": self.auto_save,
            "save_slot": self.save_slot,
            "language": self.language,
            "allow_cheats": self.allow_cheats,
            "enable_spawn_randomization": self.enable_spawn_randomization,
            "enable_weather_system": self.enable_weather_system,
            "auto_adjust_difficulty": self.auto_adjust_difficulty,
            "random_seed": self.random_seed,
            "enable_nightmare_mode": self.enable_nightmare_mode,
            "start_resources": self.start_resources,
            "resource_rate_multiplier": self.resource_rate_multiplier,
            "enemy_spawn_rate": self.enemy_spawn_rate,
            "enemy_damage_multiplier": self.enemy_damage_multiplier,
            "player_damage_multiplier": self.player_damage_multiplier,
            "enable_friendly_fire": self.enable_friendly_fire,
            "enable_auto_loot": self.enable_auto_loot,
            "loot_multiplier": self.loot_multiplier,
            "boss_health_multiplier": self.boss_health_multiplier,
            "boss_damage_multiplier": self.boss_damage_multiplier,
            "unlock_all_levels": self.unlock_all_levels,
            "unlock_all_weapons": self.unlock_all_weapons,
            "enable_rogue_like_mode": self.enable_rogue_like_mode,
            "unlock_all_buildings": self.unlock_all_buildings,
            "enable_building_upgrades": self.enable_building_upgrades,
            "ally_spawn_rate": self.ally_spawn_rate,
            "enable_save_every_level": self.enable_save_every_level,
            "enable_achievements": self.enable_achievements,
            "enable_prestige_system": self.enable_prestige_system,
            "prestige_multiplier": self.prestige_multiplier,
            "enable_leaderboards": self.enable_leaderboards,
            "enable_speedrun_timers": self.enable_speedrun_timers,
            "enable_multi_language_support": self.enable_multi_language_support,
            "enable_mod_support": self.enable_mod_support,
            "mod_folder_path": self.mod_folder_path,
            "enable_auto_update": self.enable_auto_update,
            "enable_online_features": self.enable_online_features,
            "enable_voice_chat": self.enable_voice_chat,
            "enable_cross_platform_play": self.enable_cross_platform_play,
            "enable_controller_vibration": self.enable_controller_vibration,
            "vibration_strength": self.vibration_strength,
            "max_active_quests": self.max_active_quests,
            "quest_reward_multiplier": self.quest_reward_multiplier,
            "enable_dynamic_lighting": self.enable_dynamic_lighting,
            "enable_realistic_physics": self.enable_realistic_physics,
            "physics_detail_level": self.physics_detail_level,
            "enable_procedural_generation": self.enable_procedural_generation,
            "procedural_seed": self.procedural_seed,
            "enable_fog_of_war": self.enable_fog_of_war,
            "enable_auto_farming": self.enable_auto_farming,
            "enable_customizable_difficulty": self.enable_customizable_difficulty,
            "custom_difficulty_parameters": self.custom_difficulty_parameters,
            "enable_loot_boxes": self.enable_loot_boxes,
            "enable_energy_system": self.enable_energy_system,
            "energy_system_capacity": self.energy_system_capacity,
            "energy_recharge_rate": self.energy_recharge_rate,
            "enable_weather_variations": self.enable_weather_variations,
            "enable_npc_dialogues": self.enable_npc_dialogues
        }

    def from_dict(self, data):
        self.difficulty = data.get("difficulty", self.difficulty)
        self.start_lives = data.get("start_lives", self.start_lives)
        self.max_level = data.get("max_level", self.max_level)
        self.enable_tutorial = data.get("enable_tutorial", self.enable_tutorial)
        self.auto_save = data.get("auto_save", self.auto_save)
        self.save_slot = data.get("save_slot", self.save_slot)
        self.language = data.get("language", self.language)
        self.allow_cheats = data.get("allow_cheats", self.allow_cheats)
        self.enable_spawn_randomization = data.get("enable_spawn_randomization", self.enable_spawn_randomization)
        self.enable_weather_system = data.get("enable_weather_system", self.enable_weather_system)
        self.auto_adjust_difficulty = data.get("auto_adjust_difficulty", self.auto_adjust_difficulty)
        self.random_seed = data.get("random_seed", self.random_seed)
        self.enable_nightmare_mode = data.get("enable_nightmare_mode", self.enable_nightmare_mode)
        self.start_resources = data.get("start_resources", self.start_resources)
        self.resource_rate_multiplier = data.get("resource_rate_multiplier", self.resource_rate_multiplier)
        self.enemy_spawn_rate = data.get("enemy_spawn_rate", self.enemy_spawn_rate)
        self.enemy_damage_multiplier = data.get("enemy_damage_multiplier", self.enemy_damage_multiplier)
        self.player_damage_multiplier = data.get("player_damage_multiplier", self.player_damage_multiplier)
        self.enable_friendly_fire = data.get("enable_friendly_fire", self.enable_friendly_fire)
        self.enable_auto_loot = data.get("enable_auto_loot", self.enable_auto_loot)
        self.loot_multiplier = data.get("loot_multiplier", self.loot_multiplier)
        self.boss_health_multiplier = data.get("boss_health_multiplier", self.boss_health_multiplier)
        self.boss_damage_multiplier = data.get("boss_damage_multiplier", self.boss_damage_multiplier)
        self.unlock_all_levels = data.get("unlock_all_levels", self.unlock_all_levels)
        self.unlock_all_weapons = data.get("unlock_all_weapons", self.unlock_all_weapons)
        self.enable_rogue_like_mode = data.get("enable_rogue_like_mode", self.enable_rogue_like_mode)
        self.unlock_all_buildings = data.get("unlock_all_buildings", self.unlock_all_buildings)
        self.enable_building_upgrades = data.get("enable_building_upgrades", self.enable_building_upgrades)
        self.ally_spawn_rate = data.get("ally_spawn_rate", self.ally_spawn_rate)
        self.enable_save_every_level = data.get("enable_save_every_level", self.enable_save_every_level)
        self.enable_achievements = data.get("enable_achievements", self.enable_achievements)
        self.enable_prestige_system = data.get("enable_prestige_system", self.enable_prestige_system)
        self.prestige_multiplier = data.get("prestige_multiplier", self.prestige_multiplier)
        self.enable_leaderboards = data.get("enable_leaderboards", self.enable_leaderboards)
        self.enable_speedrun_timers = data.get("enable_speedrun_timers", self.enable_speedrun_timers)
        self.enable_multi_language_support = data.get("enable_multi_language_support", self.enable_multi_language_support)
        self.enable_mod_support = data.get("enable_mod_support", self.enable_mod_support)
        self.mod_folder_path = data.get("mod_folder_path", self.mod_folder_path)
        self.enable_auto_update = data.get("enable_auto_update", self.enable_auto_update)
        self.enable_online_features = data.get("enable_online_features", self.enable_online_features)
        self.enable_voice_chat = data.get("enable_voice_chat", self.enable_voice_chat)
        self.enable_cross_platform_play = data.get("enable_cross_platform_play", self.enable_cross_platform_play)
        self.enable_controller_vibration = data.get("enable_controller_vibration", self.enable_controller_vibration)
        self.vibration_strength = data.get("vibration_strength", self.vibration_strength)
        self.max_active_quests = data.get("max_active_quests", self.max_active_quests)
        self.quest_reward_multiplier = data.get("quest_reward_multiplier", self.quest_reward_multiplier)
        self.enable_dynamic_lighting = data.get("enable_dynamic_lighting", self.enable_dynamic_lighting)
        self.enable_realistic_physics = data.get("enable_realistic_physics", self.enable_realistic_physics)
        self.physics_detail_level = data.get("physics_detail_level", self.physics_detail_level)
        self.enable_procedural_generation = data.get("enable_procedural_generation", self.enable_procedural_generation)
        self.procedural_seed = data.get("procedural_seed", self.procedural_seed)
        self.enable_fog_of_war = data.get("enable_fog_of_war", self.enable_fog_of_war)
        self.enable_auto_farming = data.get("enable_auto_farming", self.enable_auto_farming)
        self.enable_customizable_difficulty = data.get("enable_customizable_difficulty", self.enable_customizable_difficulty)
        self.custom_difficulty_parameters = data.get("custom_difficulty_parameters", self.custom_difficulty_parameters)
        self.enable_loot_boxes = data.get("enable_loot_boxes", self.enable_loot_boxes)
        self.enable_energy_system = data.get("enable_energy_system", self.enable_energy_system)
        self.energy_system_capacity = data.get("energy_system_capacity", self.energy_system_capacity)
        self.energy_recharge_rate = data.get("energy_recharge_rate", self.energy_recharge_rate)
        self.enable_weather_variations = data.get("enable_weather_variations", self.enable_weather_variations)
        self.enable_npc_dialogues = data.get("enable_npc_dialogues", self.enable_npc_dialogues)

class GameConfig:
    def __init__(self):
        self.window = WindowConfig()
        self.audio = AudioConfig()
        self.gameplay = GameplayConfig()
        self.config_file_path = "config.json"
        self.auto_save_on_exit = True
        self.enable_logging = True
        self.log_level = "info"
        self.log_file_path = "game.log"
        self.log_to_console = True
        self.enable_debug_mode = False
        self.debug_flags = []
        self.enable_profiler = False
        self.profiler_output = "profile.json"
        self.last_load_time = None
        self.last_save_time = None
        self.encryption_key = None
        self.enable_encryption = False
        self.backup_on_save = True
        self.backup_folder = "backup"
        self.enable_cloud_sync = False
        self.cloud_sync_endpoint = ""
        self.cloud_sync_token = ""
        self.enable_auto_reload = False
        self.auto_reload_interval = 60
        self.next_reload_time = 0
        self.load_error_count = 0
        self.max_load_errors = 3
        self.allow_threaded_loading = True
        self.custom_data = {}
        self.enable_localization_reload = True
        self.localization_folder = "locales"
        self.launch_args = sys.argv[1:]
        self.environment_overrides = {}
        self.enable_system_inspection = False
        self.inspection_log_path = "system_info.log"
        self.enable_patch_system = False
        self.patch_file_path = "patches.json"
        self.enable_third_party_integrations = False
        self.integration_config = {}
        self.enable_macros = False
        self.macros = {}
        self.override_player_name = None
        self.override_save_path = None
        self.enable_version_check = False
        self.version = "1.0.0"
        self.build_number = 100
        self.build_branch = "master"
        self.enable_auto_crash_reports = False
        self.crash_report_url = ""
        self.enable_telemetry = False
        self.telemetry_endpoint = ""
        self.game_uuid = str(uuid.uuid4())
        self.installation_path = str(pathlib.Path().resolve())
        self.last_update_check = None
        self.lock_file_path = "config.lock"
        self.enable_lock_file = False
        self.lock_file_handle = None
        self.enable_performance_metrics = False
        self.metrics_output = "metrics.json"
        self.metrics_interval = 30
        self.enable_advanced_graphics = False
        self.graphics_preset = "medium"
        self.enable_gpu_acceleration = True
        self.enable_shadows = True
        self.shadow_quality = "medium"
        self.texture_quality = "medium"
        self.enable_post_processing = True
        self.post_processing_quality = "medium"
        self.enable_anisotropic_filtering = True
        self.anisotropic_level = 4
        self.enable_anti_aliasing = True
        self.anti_aliasing_level = 2
        self.enable_screen_space_reflections = False
        self.enable_depth_of_field = False
        self.enable_motion_blur = False
        self.enable_bloom = False
        self.enable_color_grading = False
        self.enable_hbao = False
        self.enable_ray_tracing = False
        self.enable_upscaling = False
        self.upscaling_mode = "fsr"
        self.enable_frame_limiting = True
        self.enable_dynamic_resolution = False

    def load(self):
        if not os.path.exists(self.config_file_path):
            return
        try:
            with open(self.config_file_path, "r", encoding="utf-8") as f:
                if self.enable_encryption and self.encryption_key:
                    raw = f.read()
                    decrypted = self.decrypt(raw)
                    data = json.loads(decrypted)
                else:
                    data = json.load(f)
            self.window.from_dict(data.get("window", {}))
            self.audio.from_dict(data.get("audio", {}))
            self.gameplay.from_dict(data.get("gameplay", {}))
            self.config_file_path = data.get("config_file_path", self.config_file_path)
            self.auto_save_on_exit = data.get("auto_save_on_exit", self.auto_save_on_exit)
            self.enable_logging = data.get("enable_logging", self.enable_logging)
            self.log_level = data.get("log_level", self.log_level)
            self.log_file_path = data.get("log_file_path", self.log_file_path)
            self.log_to_console = data.get("log_to_console", self.log_to_console)
            self.enable_debug_mode = data.get("enable_debug_mode", self.enable_debug_mode)
            self.debug_flags = data.get("debug_flags", self.debug_flags)
            self.enable_profiler = data.get("enable_profiler", self.enable_profiler)
            self.profiler_output = data.get("profiler_output", self.profiler_output)
            self.last_load_time = data.get("last_load_time", self.last_load_time)
            self.last_save_time = data.get("last_save_time", self.last_save_time)
            self.encryption_key = data.get("encryption_key", self.encryption_key)
            self.enable_encryption = data.get("enable_encryption", self.enable_encryption)
            self.backup_on_save = data.get("backup_on_save", self.backup_on_save)
            self.backup_folder = data.get("backup_folder", self.backup_folder)
            self.enable_cloud_sync = data.get("enable_cloud_sync", self.enable_cloud_sync)
            self.cloud_sync_endpoint = data.get("cloud_sync_endpoint", self.cloud_sync_endpoint)
            self.cloud_sync_token = data.get("cloud_sync_token", self.cloud_sync_token)
            self.enable_auto_reload = data.get("enable_auto_reload", self.enable_auto_reload)
            self.auto_reload_interval = data.get("auto_reload_interval", self.auto_reload_interval)
            self.next_reload_time = data.get("next_reload_time", self.next_reload_time)
            self.load_error_count = data.get("load_error_count", self.load_error_count)
            self.max_load_errors = data.get("max_load_errors", self.max_load_errors)
            self.allow_threaded_loading = data.get("allow_threaded_loading", self.allow_threaded_loading)
            self.custom_data = data.get("custom_data", self.custom_data)
            self.enable_localization_reload = data.get("enable_localization_reload", self.enable_localization_reload)
            self.localization_folder = data.get("localization_folder", self.localization_folder)
            self.launch_args = data.get("launch_args", self.launch_args)
            self.environment_overrides = data.get("environment_overrides", self.environment_overrides)
            self.enable_system_inspection = data.get("enable_system_inspection", self.enable_system_inspection)
            self.inspection_log_path = data.get("inspection_log_path", self.inspection_log_path)
            self.enable_patch_system = data.get("enable_patch_system", self.enable_patch_system)
            self.patch_file_path = data.get("patch_file_path", self.patch_file_path)
            self.enable_third_party_integrations = data.get("enable_third_party_integrations", self.enable_third_party_integrations)
            self.integration_config = data.get("integration_config", self.integration_config)
            self.enable_macros = data.get("enable_macros", self.enable_macros)
            self.macros = data.get("macros", self.macros)
            self.override_player_name = data.get("override_player_name", self.override_player_name)
            self.override_save_path = data.get("override_save_path", self.override_save_path)
            self.enable_version_check = data.get("enable_version_check", self.enable_version_check)
            self.version = data.get("version", self.version)
            self.build_number = data.get("build_number", self.build_number)
            self.build_branch = data.get("build_branch", self.build_branch)
            self.enable_auto_crash_reports = data.get("enable_auto_crash_reports", self.enable_auto_crash_reports)
            self.crash_report_url = data.get("crash_report_url", self.crash_report_url)
            self.enable_telemetry = data.get("enable_telemetry", self.enable_telemetry)
            self.telemetry_endpoint = data.get("telemetry_endpoint", self.telemetry_endpoint)
            self.game_uuid = data.get("game_uuid", self.game_uuid)
            self.installation_path = data.get("installation_path", self.installation_path)
            self.last_update_check = data.get("last_update_check", self.last_update_check)
            self.lock_file_path = data.get("lock_file_path", self.lock_file_path)
            self.enable_lock_file = data.get("enable_lock_file", self.enable_lock_file)
            self.enable_performance_metrics = data.get("enable_performance_metrics", self.enable_performance_metrics)
            self.metrics_output = data.get("metrics_output", self.metrics_output)
            self.metrics_interval = data.get("metrics_interval", self.metrics_interval)
            self.enable_advanced_graphics = data.get("enable_advanced_graphics", self.enable_advanced_graphics)
            self.graphics_preset = data.get("graphics_preset", self.graphics_preset)
            self.enable_gpu_acceleration = data.get("enable_gpu_acceleration", self.enable_gpu_acceleration)
            self.enable_shadows = data.get("enable_shadows", self.enable_shadows)
            self.shadow_quality = data.get("shadow_quality", self.shadow_quality)
            self.texture_quality = data.get("texture_quality", self.texture_quality)
            self.enable_post_processing = data.get("enable_post_processing", self.enable_post_processing)
            self.post_processing_quality = data.get("post_processing_quality", self.post_processing_quality)
            self.enable_anisotropic_filtering = data.get("enable_anisotropic_filtering", self.enable_anisotropic_filtering)
            self.anisotropic_level = data.get("anisotropic_level", self.anisotropic_level)
            self.enable_anti_aliasing = data.get("enable_anti_aliasing", self.enable_anti_aliasing)
            self.anti_aliasing_level = data.get("anti_aliasing_level", self.anti_aliasing_level)
            self.enable_screen_space_reflections = data.get("enable_screen_space_reflections", self.enable_screen_space_reflections)
            self.enable_depth_of_field = data.get("enable_depth_of_field", self.enable_depth_of_field)
            self.enable_motion_blur = data.get("enable_motion_blur", self.enable_motion_blur)
            self.enable_bloom = data.get("enable_bloom", self.enable_bloom)
            self.enable_color_grading = data.get("enable_color_grading", self.enable_color_grading)
            self.enable_hbao = data.get("enable_hbao", self.enable_hbao)
            self.enable_ray_tracing = data.get("enable_ray_tracing", self.enable_ray_tracing)
            self.enable_upscaling = data.get("enable_upscaling", self.enable_upscaling)
            self.upscaling_mode = data.get("upscaling_mode", self.upscaling_mode)
            self.enable_frame_limiting = data.get("enable_frame_limiting", self.enable_frame_limiting)
            self.enable_dynamic_resolution = data.get("enable_dynamic_resolution", self.enable_dynamic_resolution)
            self.last_load_time = time.time()
        except:
            self.load_error_count += 1

    def save(self):
        if self.load_error_count >= self.max_load_errors:
            return
        data = {
            "window": self.window.to_dict(),
            "audio": self.audio.to_dict(),
            "gameplay": self.gameplay.to_dict(),
            "config_file_path": self.config_file_path,
            "auto_save_on_exit": self.auto_save_on_exit,
            "enable_logging": self.enable_logging,
            "log_level": self.log_level,
            "log_file_path": self.log_file_path,
            "log_to_console": self.log_to_console,
            "enable_debug_mode": self.enable_debug_mode,
            "debug_flags": self.debug_flags,
            "enable_profiler": self.enable_profiler,
            "profiler_output": self.profiler_output,
            "last_load_time": self.last_load_time,
            "last_save_time": self.last_save_time,
            "encryption_key": self.encryption_key,
            "enable_encryption": self.enable_encryption,
            "backup_on_save": self.backup_on_save,
            "backup_folder": self.backup_folder,
            "enable_cloud_sync": self.enable_cloud_sync,
            "cloud_sync_endpoint": self.cloud_sync_endpoint,
            "cloud_sync_token": self.cloud_sync_token,
            "enable_auto_reload": self.enable_auto_reload,
            "auto_reload_interval": self.auto_reload_interval,
            "next_reload_time": self.next_reload_time,
            "load_error_count": self.load_error_count,
            "max_load_errors": self.max_load_errors,
            "allow_threaded_loading": self.allow_threaded_loading,
            "custom_data": self.custom_data,
            "enable_localization_reload": self.enable_localization_reload,
            "localization_folder": self.localization_folder,
            "launch_args": self.launch_args,
            "environment_overrides": self.environment_overrides,
            "enable_system_inspection": self.enable_system_inspection,
            "inspection_log_path": self.inspection_log_path,
            "enable_patch_system": self.enable_patch_system,
            "patch_file_path": self.patch_file_path,
            "enable_third_party_integrations": self.enable_third_party_integrations,
            "integration_config": self.integration_config,
            "enable_macros": self.enable_macros,
            "macros": self.macros,
            "override_player_name": self.override_player_name,
            "override_save_path": self.override_save_path,
            "enable_version_check": self.enable_version_check,
            "version": self.version,
            "build_number": self.build_number,
            "build_branch": self.build_branch,
            "enable_auto_crash_reports": self.enable_auto_crash_reports,
            "crash_report_url": self.crash_report_url,
            "enable_telemetry": self.enable_telemetry,
            "telemetry_endpoint": self.telemetry_endpoint,
            "game_uuid": self.game_uuid,
            "installation_path": self.installation_path,
            "last_update_check": self.last_update_check,
            "lock_file_path": self.lock_file_path,
            "enable_lock_file": self.enable_lock_file,
            "enable_performance_metrics": self.enable_performance_metrics,
            "metrics_output": self.metrics_output,
            "metrics_interval": self.metrics_interval,
            "enable_advanced_graphics": self.enable_advanced_graphics,
            "graphics_preset": self.graphics_preset,
            "enable_gpu_acceleration": self.enable_gpu_acceleration,
            "enable_shadows": self.enable_shadows,
            "shadow_quality": self.shadow_quality,
            "texture_quality": self.texture_quality,
            "enable_post_processing": self.enable_post_processing,
            "post_processing_quality": self.post_processing_quality,
            "enable_anisotropic_filtering": self.enable_anisotropic_filtering,
            "anisotropic_level": self.anisotropic_level,
            "enable_anti_aliasing": self.enable_anti_aliasing,
            "anti_aliasing_level": self.anti_aliasing_level,
            "enable_screen_space_reflections": self.enable_screen_space_reflections,
            "enable_depth_of_field": self.enable_depth_of_field,
            "enable_motion_blur": self.enable_motion_blur,
            "enable_bloom": self.enable_bloom,
            "enable_color_grading": self.enable_color_grading,
            "enable_hbao": self.enable_hbao,
            "enable_ray_tracing": self.enable_ray_tracing,
            "enable_upscaling": self.enable_upscaling,
            "upscaling_mode": self.upscaling_mode,
            "enable_frame_limiting": self.enable_frame_limiting,
            "enable_dynamic_resolution": self.enable_dynamic_resolution
        }
        if self.backup_on_save and os.path.exists(self.config_file_path):
            if not os.path.exists(self.backup_folder):
                os.makedirs(self.backup_folder)
            backup_file = os.path.join(self.backup_folder, f"config_backup_{int(time.time())}.json")
            shutil.copy(self.config_file_path, backup_file)
        if self.enable_encryption and self.encryption_key:
            encoded = json.dumps(data)
            encrypted = self.encrypt(encoded)
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                f.write(encrypted)
        else:
            with open(self.config_file_path, "w", encoding="utf-8") as f:
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

    def apply_environment_overrides(self):
        for k, v in self.environment_overrides.items():
            os.environ[k] = str(v)

    def system_inspection(self):
        if not self.enable_system_inspection:
            return
        info = {}
        info["platform"] = platform.system()
        info["platform_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["processor"] = platform.processor()
        info["python_version"] = platform.python_version()
        info["username"] = getpass.getuser()
        info["current_dir"] = str(pathlib.Path().resolve())
        try:
            c = subprocess.check_output(["wmic", "cpu", "get", "name"], shell=True).decode()
            info["cpu_info"] = c.strip()
        except:
            info["cpu_info"] = "unknown"
        try:
            g = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", "name"], shell=True).decode()
            info["gpu_info"] = g.strip()
        except:
            info["gpu_info"] = "unknown"
        with open(self.inspection_log_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    def check_auto_reload(self):
        if not self.enable_auto_reload:
            return
        now = time.time()
        if now >= self.next_reload_time:
            self.next_reload_time = now + self.auto_reload_interval
            self.load()

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

    def patch_config(self):
        if not self.enable_patch_system:
            return
        if not os.path.exists(self.patch_file_path):
            return
        with open(self.patch_file_path, "r", encoding="utf-8") as f:
            patch_data = json.load(f)
        for key, value in patch_data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def init_random_seed(self):
        if self.gameplay.random_seed is not None:
            random.seed(self.gameplay.random_seed)
        else:
            random.seed()

    def validate_config(self):
        pass

    def start_auto_reload_thread(self):
        if not self.enable_auto_reload:
            return
        t = threading.Thread(target=self._auto_reload_loop, daemon=True)
        t.start()

    def _auto_reload_loop(self):
        while self.enable_auto_reload:
            time.sleep(1)
            self.check_auto_reload()

    def enable_macro(self, name, value):
        if not self.enable_macros:
            return
        self.macros[name] = value

    def disable_macro(self, name):
        if not self.enable_macros:
            return
        if name in self.macros:
            del self.macros[name]

    def get_macro_value(self, name):
        return self.macros.get(name, None)

    def refresh(self):
        self.check_auto_reload()
        self.system_inspection()

    def initialize(self):
        self.acquire_lock()
        self.load()
        self.apply_environment_overrides()
        self.patch_config()
        self.init_random_seed()
        self.validate_config()
        self.start_auto_reload_thread()
        if self.enable_logging and self.log_to_console:
            pass

def initialize_config():
    config = GameConfig()
    config.initialize()
    return config
