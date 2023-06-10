import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image

import pickle

from copy import deepcopy

import game

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action1', 'action2', 'next_state', 'reward1', 'reward2'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_frames, n_actions):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3*n_frames, 16, kernel_size=8, stride=2, padding=3, groups=n_frames), # (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.GELU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, groups=n_frames), # (32, 16, 16)
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1), # (32, 8, 8)
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Flatten(),

            nn.Linear(32*8*8, 256),
            nn.GELU(),

            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        # print("x.shape:", x.shape)
        out = self.main(x)
        return out

def update_state(states_, state):

    for i in range(1, n_frames):
        states_[0, (i-1)*3:i*3] = states_[0, i*3:(i+1)*3]
        # transforms.functional.to_pil_image(states_[0, (i-1)*3:i*3]).save(f"frame{i}.jpg")

    states_[0, (n_frames-1)*3:n_frames*3] = state
    # transforms.functional.to_pil_image(states_[0, (n_frames-1)*3:n_frames*3]).save(f"frame{n_frames}.jpg")

    return states_


def preprocess_state(state):
    image = Image.fromarray(state)
    image = transforms.functional.to_tensor(image)

    return image


class DQNSet:
    
    def __init__(self, number=1):

        self.policy_net = DQN(n_frames=n_frames, n_actions=n_actions).to(device)
        self.target_net = DQN(n_frames=n_frames, n_actions=n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

        self.frame_states = torch.zeros((1, 3*n_frames, 64, 64), device=device)

        self.number = number

    def select_action(self, state):
        sample = random.random()
        if sample > EPS_THRESHOLD:
            with torch.no_grad():
                out = self.policy_net(state)
                return out.max(dim=1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randint(0, n_actions-1)]], device=device, dtype=torch.long)

    def optimize_model(self):

        if len(memory) < BATCH_SIZE:
            return
        
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        if self.number == 1:
            action_batch = torch.cat(batch.action1)
            reward_batch = torch.cat(batch.reward1)
        elif self.number == 2:
            action_batch = torch.cat(batch.action2)
            reward_batch = torch.cat(batch.reward2)

        out = self.policy_net(state_batch)
        state_action_values = out.gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)

        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Huber 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # 변화도 클리핑 바꿔치기
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


if __name__ == '__main__':

    # BATCH_SIZE는 리플레이 버퍼에서 샘플링된 트랜지션의 수입니다.
    # GAMMA는 이전 섹션에서 언급한 할인 계수입니다.
    # EPS_START는 엡실론의 시작 값입니다.
    # EPS_END는 엡실론의 최종 값입니다.
    # EPS_DECAY는 엡실론의 지수 감쇠(exponential decay) 속도 제어하며, 높을수록 감쇠 속도가 느립니다.
    # TAU는 목표 네트워크의 업데이트 속도입니다.
    # LR은 ``AdamW`` 옵티마이저의 학습율(learning rate)입니다.
    BATCH_SIZE = 16
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_THRESHOLD = EPS_START
    EPS_END = 0.05
    EPS_DECAY = 0.99
    TAU = 0.005
    LR = 0.00001
    num_episodes = 1000
    n_frames = 4
    n_actions = 5
    episode_durations = []

    env = game.ShootingGame()

    state = env.reset()
    n_observations = len(state)

    memory = ReplayMemory(10000)

    reward1_list = []
    reward2_list = []

    dqn_set1 = DQNSet()
    dqn_set2 = DQNSet()

    frame_states = torch.zeros((1, 3*n_frames, 64, 64), device=device)

    for i_episode in range(num_episodes):
        print(f"EPISODE: {i_episode+1}/{num_episodes}")
        # 환경과 상태 초기화
        sum_r1 = 0
        sum_r2 = 0
        state = preprocess_state(env.reset())
        state = state.to(device).unsqueeze(0)

        # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        frame_states = update_state(deepcopy(frame_states), state)

        for t in count():
            action1 = dqn_set1.select_action(frame_states)
            action2 = dqn_set2.select_action(frame_states)

            next_state, (reward1, reward2), done, _ = env.step(action1.item(), action2.item())

            next_state = preprocess_state(next_state)
            reward1 = torch.tensor([reward1], device=device)
            reward2 = torch.tensor([reward2], device=device)
            sum_r1 += reward1.item()
            sum_r2 += reward2.item()

            if done:
                next_state = None
                frame_states = torch.zeros((1, 3*n_frames, 64, 64), device=device)
                EPS_THRESHOLD = max(EPS_THRESHOLD*EPS_DECAY, EPS_END)
                reward1_list.append(sum_r1)
                reward2_list.append(sum_r2)
                episode_durations.append(t+1)
                print(f"t: {t+1}, sum_r1: {sum_r1}, avg_r1: {round(sum_r1/t, 3)}, sum_r2: {sum_r2}, avg_r2: {round(sum_r2/t, 3)}, eps_thres: {EPS_THRESHOLD}")

            else:
                next_states = update_state(deepcopy(frame_states), next_state)
                # next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            # 메모리에 변이 저장
            memory.push(frame_states, action1, action2, next_states, reward1, reward2)

            # # 다음 상태로 이동
            # state = next_state
            frame_states = next_states

            # (정책 네트워크에서) 최적화 한단계 수행
            dqn_set1.optimize_model()
            dqn_set2.optimize_model()

            # 목표 네트워크의 가중치를 소프트 업데이트
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = dqn_set1.target_net.state_dict()
            policy_net_state_dict = dqn_set1.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            dqn_set1.target_net.load_state_dict(target_net_state_dict)


            target_net_state_dict = dqn_set2.target_net.state_dict()
            policy_net_state_dict = dqn_set2.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            dqn_set2.target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break

    torch.save(dqn_set1.policy_net.state_dict(), "./dqn_set1_policy_net.pth")
    torch.save(dqn_set2.policy_net.state_dict(), "./dqn_set2_policy_net.pth")

    train_log = {
        "reward1_list": reward1_list,
        "reward2_list": reward2_list,
        "episode_durations": episode_durations,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay": EPS_DECAY,
        "tau": TAU,
        "lr": LR,
        "num_episodes": 1000,
        "n_frames": n_frames,
        "n_actions": n_actions
    }

    pickle.dumps(train_log, "./train_log.pkl")