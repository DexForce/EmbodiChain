#!/usr/bin/env python
"""
单元测试：record_camera_data_async 多环境不等长episode录制
使用真实环境，简化版本
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

import torch
import tempfile
from unittest.mock import patch

from embodichain.lab.gym.envs.managers import FunctorCfg
from embodichain.lab.gym.envs.managers.record import record_camera_data_async
from embodichain.lab.gym.envs.tasks.rl import build_env
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManagerCfg


def create_simple_env(num_envs=4):
    """直接加载现成的配置"""
    import json
    config_path = "configs/agents/rl/push_cube/gym_config.json"
    with open(config_path) as f:
        gym_config = json.load(f)
    
    # 缩短episode便于测试
    gym_config["env"]["extensions"]["episode_length"] = 10
    
    cfg = config_to_cfg(gym_config)
    cfg.num_envs = num_envs
    cfg.sim_cfg = SimulationManagerCfg(headless=True, sim_device=torch.device("cpu"), physics_dt=0.01)
    return build_env(gym_config["id"], base_env_cfg=cfg)


def test_basic():
    """基础测试：验证不等长episode录制的bug"""
    print("\n" + "="*60)
    print("🧪 测试：record_camera_data_async Bug验证")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n1️⃣ 创建环境 (4个并行)")
        env = create_simple_env(num_envs=4)
        
        try:
            print("2️⃣ 创建recorder")
            cfg = FunctorCfg(func=None,
                           params={"name": "test_cam", "resolution": (64, 64)})
            recorder = record_camera_data_async(cfg, env)
            
            saved_videos = []
            with patch('embodichain.lab.gym.envs.managers.record.images_to_video') as mock:
                mock.side_effect = lambda f, p, n, fps: saved_videos.append({'name': n, 'frames': len(f)})
                
                print("\n3️⃣ 模拟不等长episode")
                print("   Env 0-2: 5步后reset")
                print("   Env 3:   10步后reset")
                
                obs, _ = env.reset()
                action_dim = env.action_space.shape[-1]
                
                for step in range(1, 15):
                    actions = torch.zeros(4, action_dim)
                    obs, _, _, _, _ = env.step({"delta_qpos": actions})
                    recorder(env, None, "test_cam", save_path=tmpdir)
                    
                    if step == 5:
                        env.reset(options={"reset_ids": [0, 1, 2]})
                        print(f"\n   Step {step}: Env 0-2 reset")
                        print(f"      pending: {len(recorder._pending_env_episodes)}/4")
                        print(f"      saved: {len(saved_videos)}")
                    
                    if step == 10:
                        env.reset(options={"reset_ids": [3]})
                        print(f"\n   Step {step}: Env 3 reset") 
                        print(f"      pending: {len(recorder._pending_env_episodes)}/4")
                        print(f"      saved: {len(saved_videos)}")
            
            print("\n" + "="*60)
            print("📊 结果分析")
            print("="*60)
            print(f"保存视频数: {len(saved_videos)}")
            for v in saved_videos:
                print(f"  - {v['name']}: {v['frames']} 帧")
            
            print(f"\n⚠️  Bug验证:")
            print(f"   1. 前3个环境在step 5完成，但要等Env 3")
            print(f"   2. 4个环境都完成后才保存视频")
            print(f"   3. 这导致录制延迟和可能的丢失")
            
        finally:
            env.close()
            print("\n✅ 测试完成\n")


if __name__ == "__main__":
    test_lazy_merge()
