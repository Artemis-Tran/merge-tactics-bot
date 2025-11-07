#!/usr/bin/env python3
"""
dqn_agent.py — DQN with compact 13-action space (board=6, bench=5)

- Action space (13):
    0:  Buy from hand -> next available Board slot
    1:  Buy from hand -> next available Bench slot
    2-7:   Sell Board slot i (i=0..5 mapped from current occupied board tiles in row-major order)
    8-12:  Sell Bench slot j (j=0..4)
- State vector (31 dims):
    board_occupancy[6], board_stars[6],
    bench_occupancy[5], bench_stars[5],
    hand_occupancy[6],
    [mana_norm, health_norm, round_norm]
- Rewards:
    step: +0.02 * mana - 0.01
    terminal: +10 for placement==1, +4 for placement==2, +0.1*mana bonus


Notes:
- We keep a simple local tracker for Board (20 tiles, but we present 6 "logical" slots derived
  by enumerating occupied tiles in row-major order and slicing to 6). Bench tracker is 5 slots.
- If the visual game merges automatically, our local star estimates may drift; for baseline
  learning, this is acceptable and can be tightened later with better perception.
"""

from __future__ import annotations
import argparse
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import json

import environment as env
from controller import drag, Hand, Bench, Board, play_again, return_home, start_battle
from vision import Vision
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------------
# Card and Trait Metadata
# -------------------------------------------------------------------
try:
    with open("cards.json", "r") as f:
        CARD_DATA = json.load(f)
except FileNotFoundError:
    print("Fatal: cards.json not found. Please run from the project root.")
    exit(1)

CARD_LABELS = sorted(CARD_DATA.keys())
LABEL_TO_INDEX = {label: i for i, label in enumerate(CARD_LABELS)}
NUM_CARD_TYPES = len(CARD_LABELS)

CARD_COSTS = {label: data["cost"] for label, data in CARD_DATA.items()}
CARD_TRAITS = {label: [d.get("trait1"), d.get("trait2")] for label, d in CARD_DATA.items() if d.get("trait1")}
ALL_TRAITS = sorted(list(set(t for traits in CARD_TRAITS.values() for t in traits if t)))
TRAIT_TO_INDEX = {trait: i for i, trait in enumerate(ALL_TRAITS)}
NUM_TRAITS = len(ALL_TRAITS)

# -------------------------------------------------------------------
# Layout constants
# -------------------------------------------------------------------
BOARD_ROWS = 4
BOARD_COLS = 5
BOARD_TILES = BOARD_ROWS * BOARD_COLS  # 20 physical tiles
LOGICAL_BOARD_SLOTS = 6               # we only expose 6 "sell slots"
LOGICAL_BENCH_SLOTS = 5               # we only expose 5 bench slots
MAX_HAND_SLOTS = 6

# Action space layout
NUM_BUY_ACTIONS = 1
NUM_SELL_BOARD_ACTIONS = LOGICAL_BOARD_SLOTS
NUM_SELL_BENCH_ACTIONS = LOGICAL_BENCH_SLOTS
NUM_SWAP_ACTIONS = LOGICAL_BENCH_SLOTS * LOGICAL_BOARD_SLOTS

NUM_MOVE_FRONT_ACTIONS = LOGICAL_BENCH_SLOTS  # move bench -> front row
NUM_MOVE_BACK_ACTIONS  = LOGICAL_BENCH_SLOTS  # move bench -> back row

ACTION_SIZE = (
    1
    + NUM_BUY_ACTIONS
    + NUM_SELL_BOARD_ACTIONS
    + NUM_SELL_BENCH_ACTIONS
    + NUM_SWAP_ACTIONS
    + NUM_MOVE_FRONT_ACTIONS
    + NUM_MOVE_BACK_ACTIONS
)


# State size: 6 ids + 6 stars + 5 ids + 5 stars + 6 ids + 6 upg + N_TRAITS + 4 scalars (mana, health, round, net_worth)
STATE_SIZE = (LOGICAL_BOARD_SLOTS * 2) + (LOGICAL_BENCH_SLOTS * 2) + (MAX_HAND_SLOTS * 2) + NUM_TRAITS + 4


# Helpers
def validate_state(state) -> bool:
    if getattr(state, "mana_conf", 0) < 65:
        return False

    # Count cards with both a label and decent confidence
    good_cards = sum(
        1 for c in getattr(state, "cards", [])[:6]
        if getattr(c, "label", None) and getattr(c, "conf", 0.0) >= 0.7
    )
    # Two solid reads is usually enough to proceed
    return good_cards >= 2


def logical_board_count(tracker: BoardBenchTracker) -> int:
    """How many logical board slots are currently filled (capped to 6)."""
    return min(len(tracker.list_occupied_board_indices()), LOGICAL_BOARD_SLOTS)

def is_melee(label: Optional[str]) -> bool:
    if not label: return False
    return CARD_DATA.get(label, {}).get("type", "").lower() == "melee"

def is_ranged(label: Optional[str]) -> bool:
    if not label: return False
    return CARD_DATA.get(label, {}).get("type", "").lower() == "ranged"

def save_checkpoint(agent, episode, filename):
    checkpoint = {
        "model": agent.q_online.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "episode": episode,
    }
    torch.save(checkpoint, filename)
    print(f"[save] Checkpoint saved: {filename}")


# -------------------------------------------------------------------
# Action Masking
# -------------------------------------------------------------------
def get_action_mask(game_state: env.GameState, tracker: BoardBenchTracker, recent_swap: Optional[Tuple[int, int]] = None, cap: int = 2) -> np.ndarray:
    """Returns a boolean array where True indicates a valid action."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)

    mask[0] = True

    time_left = getattr(game_state, "timer", None)
    if isinstance(time_left, (int, float)) and time_left <= 10:
        return mask

    phase = getattr(game_state, "phase", None)
    if not phase or phase not in ("deploy", "battle"):
        return mask

    # Index helpers
    BUY_BENCH_IDX = 1  # since 0 is NoOp and NUM_BUY_ACTIONS == 1

    sell_board_base = 1 + NUM_BUY_ACTIONS
    sell_bench_base = sell_board_base + NUM_SELL_BOARD_ACTIONS
    swap_base       = sell_bench_base + NUM_SELL_BENCH_ACTIONS
    move_base       = swap_base + NUM_SWAP_ACTIONS  # front then back

    num_occupied_board = len(tracker.list_occupied_board_indices())
    current_mana = int(getattr(game_state, "mana", 0) or 0)

     # === Debug Print: Shop and Mana State ===
    if DEBUG_MODE:
        print("\n[Shop Debug]")
        print(f"  Current Mana: {current_mana}")
        for i, c in enumerate(game_state.cards):
            label = getattr(c, "label", None)
            conf = getattr(c, "conf", 0.0)
            raw_cost = getattr(c, "cost", None)
            table_cost = CARD_COSTS.get(label, None)
            final_cost = table_cost if table_cost is not None else raw_cost
            print(f"  Slot {i}: {label or 'None'} | conf={conf:.2f} | "
                f"raw_cost={raw_cost} | table_cost={table_cost} | final_cost={final_cost}")
        print("-" * 60)

    def card_affordable(c):
        label = getattr(c, "label", None)
        if not label:
            return False
        return CARD_COSTS.get(label, 9999) <= current_mana
    
    debug_cards = [
        (getattr(c, "label", None), CARD_COSTS.get(getattr(c, "label", None), None))
        for c in game_state.cards[:MAX_HAND_SLOTS]
    ]
    print(f"[mask] mana={current_mana} hand={debug_cards}")
    can_buy = any(card_affordable(c) for c in game_state.cards[:MAX_HAND_SLOTS])

    if phase == "battle":
        # --- BATTLE: Only bench-safe actions ---
        # Buy -> Bench
        if can_buy and sum(1 for s in tracker.bench_stars if s > 0) < LOGICAL_BENCH_SLOTS:
            mask[BUY_BENCH_IDX] = True

        # Sell Bench
        for j in range(LOGICAL_BENCH_SLOTS):
            if tracker.bench_stars[j] > 0:
                mask[sell_bench_base + j] = True

        return mask

    # --- DEPLOY: bench buy is allowed; board buy is removed ---
    if can_buy and sum(1 for s in tracker.bench_stars if s > 0) < LOGICAL_BENCH_SLOTS:
        mask[BUY_BENCH_IDX] = True

    # Sell Board
    for i in range(LOGICAL_BOARD_SLOTS):
        if i < num_occupied_board:
            mask[sell_board_base + i] = True

    # Sell Bench
    for j in range(LOGICAL_BENCH_SLOTS):
        if tracker.bench_stars[j] > 0:
            mask[sell_bench_base + j] = True

    # Swaps (deploy only)
    swap_base = NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS
    for bench_idx in range(LOGICAL_BENCH_SLOTS):
        if tracker.bench_stars[bench_idx] == 0:
            continue  # empty bench slot can't be swapped

        for board_logical_idx in range(LOGICAL_BOARD_SLOTS):
            k = swap_base + bench_idx * LOGICAL_BOARD_SLOTS + board_logical_idx
            mask[k] = False

            board_has_unit = tracker.board_stars[board_logical_idx] > 0
            if not board_has_unit:
                continue

            # block the immediately repeated exact same swap pair
            if recent_swap == (bench_idx, board_logical_idx):
                continue

            mask[k] = True

    # Move Bench -> Front/Back Row
    board_count = sum(1 for s in tracker.board_stars if s > 0)
    under_cap = board_count < cap
   
    for bench_idx in range(LOGICAL_BENCH_SLOTS):
        if tracker.bench_stars[bench_idx] == 0:
            continue

        label = tracker.bench_labels[bench_idx]
        front_idx = move_base + bench_idx
        back_idx  = move_base + LOGICAL_BENCH_SLOTS + bench_idx

        mask[front_idx] = False
        mask[back_idx]  = False

        if under_cap:
            if is_melee(label):
                mask[front_idx] = True
            if is_ranged(label):
                mask[back_idx] = True
    return mask


def is_board_affecting_action(action_index: int) -> bool:
    """
    Returns True if the action touches the board (illegal in battle):
      - Buy->Board
      - Sell Board
      - Swaps
    Bench-only (Buy->Bench, Sell Bench) returns False.
    """
    if action_index <= 0:
        return False
    a = action_index - 1
    # Buy actions (0: to board, 1: to bench)
    if a < NUM_BUY_ACTIONS:
        return False
    a -= NUM_BUY_ACTIONS

    # Sell Board
    if a < NUM_SELL_BOARD_ACTIONS:
        return True
    a -= NUM_SELL_BOARD_ACTIONS

    # Sell Bench
    if a < NUM_SELL_BENCH_ACTIONS:
        return False
    a -= NUM_SELL_BENCH_ACTIONS

    # Swaps
    if a < NUM_SWAP_ACTIONS:
        return True
    a -= NUM_SWAP_ACTIONS

    # Move Bench -> Front
    if a < NUM_MOVE_FRONT_ACTIONS:
        return True
    a -= NUM_MOVE_FRONT_ACTIONS

    # Move Bench -> Back
    if a < NUM_MOVE_BACK_ACTIONS:
        return True

    # Fallback: unknown region
    return False




# -------------------------------------------------------------------
# Net Worth Calculation
# -------------------------------------------------------------------
def calculate_net_worth(tracker: BoardBenchTracker, current_mana: int) -> Tuple[int, int]:
    """Calculates total value of units on board/bench and net worth."""
    total_unit_value = 0
    for i in range(len(tracker.board_labels)):
        label = tracker.board_labels[i]
        stars = tracker.board_stars[i]
        if label and stars > 0:
            total_unit_value += CARD_COSTS.get(label, 0) * stars

    for i in range(len(tracker.bench_labels)):
        label = tracker.bench_labels[i]
        stars = tracker.bench_stars[i]
        if label and stars > 0:
            total_unit_value += CARD_COSTS.get(label, 0) * stars
            
    return total_unit_value, total_unit_value + current_mana

def calculate_board_value(tracker: BoardBenchTracker) -> int:
    """Total gold-equivalent of units currently deployed on the board (ignores bench)."""
    total = 0
    for i, stars in enumerate(tracker.board_stars):
        if stars > 0:
            label = tracker.board_labels[i]
            total += CARD_COSTS.get(label, 0) * stars
    return total

def trait_levels_for_tracker(tracker: BoardBenchTracker) -> Tuple[int, int]:
    """Return (#level1_traits, #level2_traits) using the same thresholds as step()."""
    trait_counts = {trait: 0 for trait in ALL_TRAITS}
    for label in tracker.board_labels:
        if label in CARD_TRAITS:
            for t in CARD_TRAITS[label]:
                if t in trait_counts:
                    trait_counts[t] += 1
    num_level1 = sum(1 for c in trait_counts.values() if 2 <= c < 4)
    num_level2 = sum(1 for c in trait_counts.values() if c >= 4)
    return num_level1, num_level2


# -------------------------------------------------------------------
# Lightweight trackers
# -------------------------------------------------------------------
@dataclass
class BoardBenchTracker:
    """
    Tracks what's on the Board (20 tiles) and Bench (5 slots).
    Now includes card costs and handles optimistic merge updates.
    """
    board_labels: List[Optional[str]]
    board_stars: List[int]
    bench_labels: List[Optional[str]]
    bench_stars: List[int]

    @classmethod
    def fresh(cls) -> "BoardBenchTracker":
        return cls(board_labels=[None]*BOARD_TILES,
                   board_stars=[0]*BOARD_TILES,
                   bench_labels=[None]*LOGICAL_BENCH_SLOTS,
                   bench_stars=[0]*LOGICAL_BENCH_SLOTS)

    def list_occupied_board_indices(self) -> List[int]:
        return [idx for idx, star in enumerate(self.board_stars) if star > 0]

    def first_empty_board_index(self) -> Optional[int]:
        for idx, star in enumerate(self.board_stars):
            if star == 0:
                return idx
        return None

    def first_empty_bench_slot(self) -> Optional[int]:
        for j in range(LOGICAL_BENCH_SLOTS):
            if self.bench_stars[j] == 0:
                return j
        return None

    def find_and_upgrade(self, label: str) -> bool:
        """Finds a unit of the same label and lowest star level to upgrade it."""
        # Prioritize upgrading units on the bench first
        for i in range(LOGICAL_BENCH_SLOTS):
            if self.bench_labels[i] == label and self.bench_stars[i] < 3:
                self.bench_stars[i] += 1
                return True
        # Then check the board
        for i in range(BOARD_TILES):
            if self.board_labels[i] == label and self.board_stars[i] < 3:
                self.board_stars[i] += 1
                return True
        return False

    def set_board_tile(self, index: int, label: Optional[str], star_level: int):
        self.board_labels[index] = label
        self.board_stars[index] = int(max(0, min(3, star_level)))

    def clear_board_tile(self, index: int):
        self.board_labels[index] = None
        self.board_stars[index] = 0

    def set_bench_slot(self, slot: int, label: Optional[str], star_level: int):
        self.bench_labels[slot] = label
        self.bench_stars[slot] = int(max(0, min(3, star_level)))

    def clear_bench_slot(self, slot: int):
        self.bench_labels[slot] = None
        self.bench_stars[slot] = 0

# -------------------------------------------------------------------
# State encoder
# -------------------------------------------------------------------
def vectorize_state(game_state: env.GameState, tracker: BoardBenchTracker) -> np.ndarray:
    """
    Build the state vector with card identities, properties, and active traits.
    """
    # --- Trait Activation (from board units only) ---
    trait_counts = {trait: 0 for trait in ALL_TRAITS}
    board_unit_labels = [label for label in tracker.board_labels if label is not None]
    for label in board_unit_labels:
        if label in CARD_TRAITS:
            for trait in CARD_TRAITS[label]:
                if trait in trait_counts:
                    trait_counts[trait] += 1

    trait_activations = np.zeros(NUM_TRAITS, dtype=np.float32)
    for i, trait in enumerate(ALL_TRAITS):
        count = trait_counts[trait]
        if count >= 4:
            trait_activations[i] = 1.0  # Level 2 bonus
        elif count >= 2:
            trait_activations[i] = 0.5  # Level 1 bonus

    # --- Board State ---
    occupied_indices = tracker.list_occupied_board_indices()
    board_slot_indices = (occupied_indices + [-1]*LOGICAL_BOARD_SLOTS)[:LOGICAL_BOARD_SLOTS]

    board_identities = np.zeros(LOGICAL_BOARD_SLOTS, dtype=np.float32)
    board_star_levels = np.zeros(LOGICAL_BOARD_SLOTS, dtype=np.float32)
    for i, idx in enumerate(board_slot_indices):
        if idx >= 0:
            label = tracker.board_labels[idx]
            if label in LABEL_TO_INDEX:
                board_identities[i] = (LABEL_TO_INDEX[label] + 1) / NUM_CARD_TYPES
            board_star_levels[i] = tracker.board_stars[idx] / 3.0

    # --- Bench State ---
    bench_identities = np.zeros(LOGICAL_BENCH_SLOTS, dtype=np.float32)
    bench_star_levels = np.zeros(LOGICAL_BENCH_SLOTS, dtype=np.float32)
    for j in range(LOGICAL_BENCH_SLOTS):
        if tracker.bench_stars[j] > 0:
            label = tracker.bench_labels[j]
            if label in LABEL_TO_INDEX:
                bench_identities[j] = (LABEL_TO_INDEX[label] + 1) / NUM_CARD_TYPES
            bench_star_levels[j] = tracker.bench_stars[j] / 3.0

    # --- Hand State ---
    hand_identities = np.zeros(MAX_HAND_SLOTS, dtype=np.float32)
    hand_upgradable = np.zeros(MAX_HAND_SLOTS, dtype=np.float32)
    for h_idx, card in enumerate(game_state.cards[:MAX_HAND_SLOTS]):
        label = getattr(card, "label", None)
        if label and getattr(card, "conf", 0.0) >= 0.4:
            if label in LABEL_TO_INDEX:
                hand_identities[h_idx] = (LABEL_TO_INDEX[label] + 1) / NUM_CARD_TYPES
            if getattr(card, "upgradable", False):
                hand_upgradable[h_idx] = 1.0

    # --- Scalar State (handle None) ---
    raw_mana = getattr(game_state, "mana", 0.0)
    current_mana = int(raw_mana if raw_mana is not None else 0.0)
    mana_norm = float(current_mana) / 20.0

    raw_health = getattr(game_state, "health", 0.0)
    health_norm = float(raw_health if raw_health is not None else 0.0) / 100.0

    raw_round = getattr(game_state, "round", 0.0)
    round_norm = float(raw_round if raw_round is not None else 0.0) / 30.0
    
    _, net_worth = calculate_net_worth(tracker, current_mana)
    net_worth_norm = net_worth / 100.0 # Normalize by a reasonable late-game value

    scalar_block = np.array([mana_norm, health_norm, round_norm, net_worth_norm], dtype=np.float32)

    # --- Concatenate ---
    state_vector = np.concatenate([
        board_identities, board_star_levels,
        bench_identities, bench_star_levels,
        hand_identities, hand_upgradable,
        trait_activations,
        scalar_block
    ], axis=0)

    assert state_vector.shape[0] == STATE_SIZE, f"Expected STATE_SIZE {STATE_SIZE}, got {state_vector.shape[0]}"
    return state_vector

# -------------------------------------------------------------------
# Q-network
# -------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# -------------------------------------------------------------------
# Replay Buffer
# -------------------------------------------------------------------
@dataclass
class Transition:
    state_vector: np.ndarray
    action_index: int
    reward_value: float
    next_state_vector: np.ndarray
    done_flag: bool

class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.write_pointer = 0

    def push(self, transition: Transition):
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.write_pointer] = transition
        self.write_pointer = (self.write_pointer + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[i] for i in indices]

    def __len__(self):
        return len(self.storage)

# -------------------------------------------------------------------
# DQN Agent
# -------------------------------------------------------------------
class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 gamma: float = 0.98,
                 learning_rate: float = 3e-4,
                 target_tau: float = 0.01):
        self.q_online = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q_target = QNetwork(state_dim, action_dim).to(DEVICE)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.optimizer = optim.Adam(self.q_online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.target_tau = target_tau
        self.max_desired_board_seen = 2

    @torch.no_grad()
    def act(self, state_vector: np.ndarray, epsilon: float, mask: np.ndarray) -> int:
        if not np.any(mask):
            return -1 # Sentinel for no-op

        if random.random() < epsilon:
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            print(f"  - Exploring: Taking random valid action {action}")
            return action
        
        input_tensor = torch.from_numpy(state_vector).float().to(DEVICE).unsqueeze(0)
        q_values = self.q_online(input_tensor).squeeze(0)
        
        # Apply mask
        q_values[~mask] = -float('inf')
        
        action = int(torch.argmax(q_values).item())
        
        q_values_str = ", ".join([f"{q:.2f}" for q in q_values])
        print(f"  - Exploiting: Q-Values: [{q_values_str}] -> Action {action}")
        return action

    def train_step(self, batch: List[Transition]) -> float:
        state_batch = torch.tensor(np.stack([b.state_vector for b in batch]), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor([b.action_index for b in batch], dtype=torch.long, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor([b.reward_value for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state_batch = torch.tensor(np.stack([b.next_state_vector for b in batch]), dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor([1.0 if b.done_flag else 0.0 for b in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_selected = self.q_online(state_batch).gather(1, action_batch)  # [B,1]

        with torch.no_grad():
            max_next = torch.max(self.q_target(next_state_batch), dim=1, keepdim=True)[0]
            target_values = reward_batch + (1.0 - done_batch) * self.gamma * max_next

        loss = nn.functional.smooth_l1_loss(q_selected, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_online.parameters(), 1.0)
        self.optimizer.step()

        # Soft update of target network
        with torch.no_grad():
            for param_online, param_target in zip(self.q_online.parameters(), self.q_target.parameters()):
                param_target.data.mul_(1.0 - self.target_tau).add_(self.target_tau * param_online.data)

        return float(loss.item())

# -------------------------------------------------------------------
# Environment wrapper to execute 13 actions
# -------------------------------------------------------------------
@dataclass
class MergeTacticsEnv:
    vision: Vision
    tracker: BoardBenchTracker
    last_swap_pair: Optional[Tuple[int, int]] = None

    def reset(self) -> Tuple[np.ndarray, env.GameState]:
        self.tracker = BoardBenchTracker.fresh()
        self.max_desired_board_seen = 2
        self.last_swap_pair = None

        max_rf_attempts = 5         # limit roboflow attempts per cycle
        max_play_again_attempts = 2 # try play_again() up to twice

        play_again_attempts = 0
        label, tile_index = None, None
        valid_state = False

        while not valid_state or not label:
            for attempt in range(max_rf_attempts):
                frame = self.vision.capture_frame()
                state = env.get_state(frame)
                valid_state = validate_state(state)
                if not valid_state:
                    continue
                label, tile_index = env.get_roboflow_prediction(frame)
                if label:
                    break  # success
                time.sleep(0.3)

            # if still no label after several tries, try to reset game
            if not label:
                play_again_attempts += 1
                print(f"[reset] No valid prediction after {max_rf_attempts} attempts "
                    f"(attempt {play_again_attempts}/{max_play_again_attempts}). Trying play_again()...")
                play_again()
                time.sleep(2)

                if play_again_attempts >= max_play_again_attempts:
                    print("[reset] Still stuck after play_again retries. Clearing board manually...")
                    try:
                        drag(Board(0, 2), Hand(0))
                        drag(Board(3, 2), Hand(0))
                    except Exception as e:
                        print(f"[reset] Manual clear failed: {e}")
                    break  # stop looping — proceed anyway

        label = label or "Unknown"
        tile_index = tile_index or 0

        if label != "Unknown" and tile_index is not None:
            print(f"Roboflow predicted {label} at index {tile_index}")
            self.tracker.set_board_tile(tile_index, label, star_level=1)
        else:
            print("[reset] No valid unit detected — board assumed empty.")

        game_state = env.get_state(frame)
        state_vector = vectorize_state(game_state, self.tracker)
        return state_vector, game_state


    # ----------------------- helpers -----------------------
    def _enumerate_board_occupied_indices(self) -> List[int]:
        return self.tracker.list_occupied_board_indices()
    
    def _bench_is_full(self) -> bool:
        return sum(1 for s in self.tracker.bench_stars if s > 0) >= LOGICAL_BENCH_SLOTS
    
    def compute_desired_board_for_state(self, game_state) -> int:
        # raw desired from the current round
        round_num = int(getattr(game_state, "round", 0) or 0)
        raw_desired = min(round_num + 1, LOGICAL_BOARD_SLOTS)

        # make it monotonic for this episode
        if raw_desired > self.max_desired_board_seen:
            self.max_desired_board_seen = raw_desired

        return self.max_desired_board_seen
    
    def _first_empty_col_in_row(self, row: int) -> Optional[int]:
        start = row * BOARD_COLS
        for c in range(BOARD_COLS):
            if self.tracker.board_stars[start + c] == 0:
                return c
        return None

    def _move_bench_to_row(self, bench_idx: int, row: int):
        """
        Move a bench unit to the given board row.
        - Prefer the first empty tile in that row.
        - If the target tile is occupied, perform a swap with that tile.
        """
        if not (0 <= bench_idx < LOGICAL_BENCH_SLOTS):
            return
        if self.tracker.bench_stars[bench_idx] == 0:
            return

        # Choose a column: first empty if available, else bench_idx % BOARD_COLS
        col = self._first_empty_col_in_row(row)
        if col is None:
            col = bench_idx % BOARD_COLS

        target_phys = row * BOARD_COLS + col

        # Execute drag
        drag(Bench(bench_idx), Board(row, col))

        bench_label = self.tracker.bench_labels[bench_idx]
        bench_stars = self.tracker.bench_stars[bench_idx]
        board_label = self.tracker.board_labels[target_phys]
        board_stars = self.tracker.board_stars[target_phys]

        if board_stars > 0:
            # Swap semantics
            self.tracker.set_board_tile(target_phys, bench_label, bench_stars)
            self.tracker.set_bench_slot(bench_idx, board_label, board_stars)
            print(f"  - Moved (swap) Bench {bench_idx} ({bench_label} ★{bench_stars}) ↔ Board ({row},{col}) ({board_label} ★{board_stars})")
        else:
            # Simple move
            self.tracker.set_board_tile(target_phys, bench_label, bench_stars)
            self.tracker.clear_bench_slot(bench_idx)
            print(f"  - Moved Bench {bench_idx} ({bench_label} ★{bench_stars}) → Board ({row},{col})")

    def _move_bench_to_front(self, bench_idx: int):
        """Move a bench unit to the front row (row 0)."""
        self._move_bench_to_row(bench_idx, row=0)

    def _move_bench_to_back(self, bench_idx: int):
        """Move a bench unit to the back row (row 3)."""
        self._move_bench_to_row(bench_idx, row=BOARD_ROWS - 1)


    def _map_logical_board_slot_to_physical_index(self, logical_index: int) -> Optional[int]:
        occupied = self._enumerate_board_occupied_indices()
        if logical_index < len(occupied):
            return occupied[logical_index]
        return None

    def _first_nonempty_hand_index(self, game_state: env.GameState) -> Optional[int]:
        for h_idx, card in enumerate(game_state.cards[:MAX_HAND_SLOTS]):
            label = getattr(card, "label", None)
            conf = getattr(card, "conf", 0.0)
            if label and conf >= 0.3:
                return h_idx
        # Fallback: if any raw text, still try
        for h_idx, card in enumerate(game_state.cards[:MAX_HAND_SLOTS]):
            if getattr(card, "label", None):
                return h_idx
        return None

    def _sell_board_slot(self, logical_slot: int):
        physical_idx = self._map_logical_board_slot_to_physical_index(logical_slot)
        if physical_idx is None:
            return
        row, col = divmod(physical_idx, BOARD_COLS)
        drag(Board(row, col), Hand(0))
        self.tracker.clear_board_tile(physical_idx)

    def _sell_bench_slot(self, bench_slot: int):
        if bench_slot < 0 or bench_slot >= LOGICAL_BENCH_SLOTS:
            return
        if self.tracker.bench_stars[bench_slot] == 0:
            return
        drag(Bench(bench_slot), Hand(0))
        self.tracker.clear_bench_slot(bench_slot)

    def _buy_from_hand(self, hand_index: int, card: env.Card, to_board: bool = False) -> Tuple[bool, bool]:
        """Handles buying a card to board or bench, with merge logic.
        Returns (merged, bought)."""
        merged = False
        bought = False

        if getattr(card, "upgradable", False) and card.label:
            # This card will cause a merge. The game handles the merge itself when we buy it.
            drag(Hand(hand_index), Bench(4))  # Drag to a bench slot to buy
            self.tracker.find_and_upgrade(card.label)
            merged = True
            bought = True
        else:
            if to_board:
                if sum(1 for s in self.tracker.board_stars if s > 0) >= LOGICAL_BOARD_SLOTS:
                    return merged, bought  # logical cap reached
                board_idx = self.tracker.first_empty_board_index()
                if board_idx is not None:
                    row, col = divmod(board_idx, BOARD_COLS)
                    drag(Hand(hand_index), Board(row, col))
                    self.tracker.set_board_tile(board_idx, label=card.label, star_level=1)
                    bought = True
            else:  # to bench
                if sum(1 for s in self.tracker.bench_stars if s > 0) >= LOGICAL_BENCH_SLOTS:
                    return merged, bought
                bench_slot = self.tracker.first_empty_bench_slot()
                if bench_slot is not None:
                    drag(Hand(hand_index), Bench(bench_slot))
                    self.tracker.set_bench_slot(bench_slot, label=card.label, star_level=1)
                    bought = True

        return merged, bought


    def _swap_units(self, bench_idx: int, board_logical_idx: int):
        """Swaps a unit from the bench with a unit on the board."""
        board_physical_idx = self._map_logical_board_slot_to_physical_index(board_logical_idx)
        if board_physical_idx is None:
            return

        # Perform the drag
        row, col = divmod(board_physical_idx, BOARD_COLS)
        drag(Bench(bench_idx), Board(row, col))

        # Optimistically update the tracker
        board_label = self.tracker.board_labels[board_physical_idx]
        board_stars = self.tracker.board_stars[board_physical_idx]
        bench_label = self.tracker.bench_labels[bench_idx]
        bench_stars = self.tracker.bench_stars[bench_idx]

        self.tracker.set_board_tile(board_physical_idx, bench_label, bench_stars)
        self.tracker.set_bench_slot(bench_idx, board_label, board_stars)
        print(f"  - Swapped bench slot {bench_idx} ({bench_label}) with board slot {board_logical_idx} ({board_label})")

    def _terminal_outcome(self, next_game_state):
        game_over, will_play_again = getattr(next_game_state, "game_over", (False, False))
        placement_value = None
        terminal_reward = 0.0
        if game_over:
            time.sleep(1)
            end_frame = self.vision.capture_frame()
            placement = env.get_placement(end_frame)
            placement_value = int(placement) if placement else 8
            if placement_value == 1:
                terminal_reward = 50.0
            elif placement_value == 2:
                terminal_reward = 25.0
            elif placement_value == 4:
                terminal_reward = -25.0
        return game_over, bool(will_play_again), placement_value, terminal_reward


    # ----------------------- core step -----------------------
    def step(self, action_index: int, last_game_state: env.GameState, forced_noop: bool = False) -> Tuple[np.ndarray, env.GameState, float, bool, dict]:
        """
        Executes an action and observes the next state and reward.
        """
        if forced_noop:
            frame = self.vision.capture_frame()
            next_game_state = env.get_state(frame)
            next_state_vector = vectorize_state(next_game_state, self.tracker)

            done_flag, will_play_again, placement_value, terminal_reward = self._terminal_outcome(next_game_state)

            info = {
                "forced_noop_tick": True,
                "placement": placement_value,
                "play_again": will_play_again,
            }
            reward_value = 0.0 + terminal_reward
            return next_state_vector, next_game_state, reward_value, done_flag, info
                
        # Snapshot tracker so we can roll back if UI didn't commit the action.
        tracker_snapshot = copy.deepcopy(self.tracker)
        pre_state_vector = vectorize_state(last_game_state, self.tracker)

        bench_full_pre = self._bench_is_full()
        sold_one_star = False
        
        prev_mana = int(getattr(last_game_state, "mana", 0.0) or 0.0)
        prev_total_unit_value, _ = calculate_net_worth(tracker_snapshot, 0)

        prev_board_slots = logical_board_count(tracker_snapshot)      
        prev_board_value = calculate_board_value(tracker_snapshot)    
        prev_l1, prev_l2 = trait_levels_for_tracker(tracker_snapshot) 

        merged = False
        bought = False
        last_phase = getattr(last_game_state, "phase", None)
        if last_phase == "battle" and is_board_affecting_action(action_index):
            print(f"[step] Phase=battle → blocking board-affecting action {action_index}; NoOp instead")
            action_index = 0
        action_desc = "NoOp"

        # --- Action Decoding ---
        if action_index == 0:
            pass  # Action 0 is No-op
        else:
            action = action_index - 1 
            if action < NUM_BUY_ACTIONS:  # Only: Buy -> Bench
                hand_index = self._first_nonempty_hand_index(last_game_state)
                if hand_index is not None:
                    card = last_game_state.cards[hand_index]
                    label = getattr(card, "label", None)
                    mana = int(getattr(last_game_state, "mana", 0) or 0)
                    if label and CARD_COSTS.get(label, 9999) <= mana:
                        action_desc = f"Buy {label} to Bench"
                        merged, bought = self._buy_from_hand(hand_index, card)


            elif action < NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS:
                board_slot = action - NUM_BUY_ACTIONS
                # Look up the physical tile to read stars BEFORE we sell
                phys = self._map_logical_board_slot_to_physical_index(board_slot)
                stars_pre = self.tracker.board_stars[phys] if phys is not None else 0
                sold_one_star = (stars_pre == 1)
                action_desc = f"Sell Board Slot {board_slot}"
                self._sell_board_slot(board_slot)

            elif action < NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS:
                bench_slot = action - (NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS)
                stars_pre = self.tracker.bench_stars[bench_slot] if 0 <= bench_slot < LOGICAL_BENCH_SLOTS else 0
                sold_one_star = (stars_pre == 1)
                action_desc = f"Sell Bench Slot {bench_slot}"
                self._sell_bench_slot(bench_slot)
            elif action < NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS + NUM_SWAP_ACTIONS:
                tmp = action - (NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS)
                bench_idx = tmp // LOGICAL_BOARD_SLOTS
                board_logical_idx = tmp % LOGICAL_BOARD_SLOTS
                action_desc = f"Swap Bench {bench_idx} ↔ Board {board_logical_idx}"
                self._swap_units(bench_idx, board_logical_idx)
                self.last_swap_pair = (bench_idx, board_logical_idx)
            else:
                # New movement actions region
                base_idx = action - (
                    NUM_BUY_ACTIONS
                    + NUM_SELL_BOARD_ACTIONS
                    + NUM_SELL_BENCH_ACTIONS
                    + NUM_SWAP_ACTIONS
                )
                if base_idx < NUM_MOVE_FRONT_ACTIONS:
                    bench_idx = base_idx
                    action_desc = f"Move Bench {bench_idx} → Front Row"
                    self._move_bench_to_front(bench_idx)
                else:
                    base_idx -= NUM_MOVE_FRONT_ACTIONS
                    if base_idx < NUM_MOVE_BACK_ACTIONS:
                        bench_idx = base_idx
                        action_desc = f"Move Bench {bench_idx} → Back Row"
                        self._move_bench_to_back(bench_idx)
                    else:
                        # Safety fallback (shouldn't happen)
                        action_desc = "NoOp (unknown action region)"

        print(f"Action: {action_desc}")
        if not action_desc.startswith("Swap"):
            self.last_swap_pair = None

        # Observe next state
        time.sleep(0.4) # Allow UI to update
        frame = self.vision.capture_frame()
        next_game_state = env.get_state(frame)

        phase = getattr(next_game_state, "phase", None)
        mana_conf = getattr(next_game_state, "mana_conf", 0.0)
        game_over, will_play_again  = getattr(next_game_state, "game_over", (False, False))

        if (mana_conf < 75) and (not game_over):
            print("[step] rolling back tracker and treating as NoOp.")
            self.tracker = tracker_snapshot
            return pre_state_vector, last_game_state, 0.0, False, {"rolled_back": True}

        occupied_board = [
            (idx, self.tracker.board_labels[idx], self.tracker.board_stars[idx])
            for idx in range(BOARD_TILES)
            if self.tracker.board_stars[idx] > 0
        ]
        occupied_bench = [
            (j, self.tracker.bench_labels[j], self.tracker.bench_stars[j])
            for j in range(LOGICAL_BENCH_SLOTS)
            if self.tracker.bench_stars[j] > 0
        ]

        print("\n[Tracker State]")
        print("  Board occupied tiles:")
        if occupied_board:
            for idx, label, stars in occupied_board:
                row, col = divmod(idx, BOARD_COLS)
                print(f"    - ({row},{col}) idx={idx}: {label or 'None'} ★{stars}")
        else:
            print("    (empty)")

        print("  Bench occupied slots:")
        if occupied_bench:
            for j, label, stars in occupied_bench:
                print(f"    - slot {j}: {label or 'None'} ★{stars}")
        else:
            print("    (empty)")

        # --- Trait State Tracking -------------------------------------------------
        trait_counts = {trait: 0 for trait in ALL_TRAITS}
        for label in self.tracker.board_labels:
            if label in CARD_TRAITS:
                for t in CARD_TRAITS[label]:
                    if t in trait_counts:
                        trait_counts[t] += 1

        print("  Active Traits:")
        active_any = False
        for trait, count in trait_counts.items():
            if count > 0:
                active_any = True
                level = "Level 2" if count >= 4 else "Level 1" if count >= 2 else "Base"
                print(f"    - {trait}: {count} ({level})")
        if not active_any:
            print("    (none)")

        num_level1 = 0
        num_level2 = 0
        for trait, count in trait_counts.items():
            if count >= 4:
                num_level2 += 1
            elif count >= 2:
                num_level1 += 1

        # --- Reward Shaping ---
        current_mana = int(getattr(next_game_state, "mana", 0.0) or 0.0)
        total_unit_value, _ = calculate_net_worth(self.tracker, 0)
        
        LAMBDA_MANA = 0.5    # how much to value unspent mana vs units
        NET_K = 0.03         # scale for the delta-net-worth reward

        merge_reward = 10 if merged else 0.0
        if merged:
            print("  - MERGE DETECTED! +10 reward")

        # Compute current values AFTER the action/UI settles:
        current_mana = int(getattr(next_game_state, "mana", 0.0) or 0.0)
        total_unit_value, _ = calculate_net_worth(self.tracker, 0)
        curr_net = total_unit_value + LAMBDA_MANA * current_mana
        prev_net = prev_total_unit_value + LAMBDA_MANA * prev_mana
        networth_reward = NET_K * (curr_net - prev_net)

        ALPHA_SLOTS = 0.6
        curr_board_slots = logical_board_count(self.tracker)
        board_slots_delta_reward = ALPHA_SLOTS * float(curr_board_slots - prev_board_slots)

        DELTA_L1_W = 0.4
        DELTA_L2_W = 0.7
        curr_l1, curr_l2 = trait_levels_for_tracker(self.tracker)
        trait_delta_reward = 0.0
        phase = getattr(next_game_state, "phase", None)
        if phase == "battle":
            trait_delta_reward = DELTA_L1_W * (curr_l1 - prev_l1) + DELTA_L2_W * (curr_l2 - prev_l2)

        DELTA_BOARD_VALUE_K = 0.01
        curr_board_value = calculate_board_value(self.tracker)
        board_value_delta_reward = 0.0
        if phase == "battle":
            board_value_delta_reward = DELTA_BOARD_VALUE_K * (curr_board_value - prev_board_value)

        BUY_COMMIT_REWARD = 0.2
        buy_reward = BUY_COMMIT_REWARD if (bought and (curr_net - prev_net) > 0) else 0.0

        SELL_1STAR_PENALTY = -1
        sell_penalty = SELL_1STAR_PENALTY if (sold_one_star and not bench_full_pre) else 0.0

        pointless_swap_penalty = 0.0
        if action_desc.startswith("Swap"):
            no_progress = (
                abs(networth_reward) < 1e-6 and
                abs(board_slots_delta_reward) < 1e-6 and
                abs(trait_delta_reward) < 1e-6 and
                abs(board_value_delta_reward) < 1e-6
            )
            if no_progress:
                pointless_swap_penalty = -0.05

        tick_penalty = -0.01

        step_reward = (
            networth_reward
            + merge_reward
            + board_slots_delta_reward
            + trait_delta_reward
            + board_value_delta_reward
            + buy_reward
            + sell_penalty
            + pointless_swap_penalty
            + tick_penalty
        )

        print(
            f"  - Reward: {step_reward:.3f} "
            f"(ΔNet: {networth_reward:.3f}, Merge: {merge_reward:.3f}, "
            f"ΔSlots: {board_slots_delta_reward:.3f}, ΔTraits: {trait_delta_reward:.3f}, "
            f"ΔBoardVal: {board_value_delta_reward:.3f}, Buy: {buy_reward:.3f}, Sell: {sell_penalty:.3f} "
            f"| phase={phase} round={getattr(next_game_state, 'round', 0)})"
        )
        print("==============================================================================================\n")
        done_flag= False
        next_state_vector = vectorize_state(next_game_state, self.tracker)
        info = {"play_again": bool(will_play_again)}
        return next_state_vector, next_game_state, float(step_reward), done_flag, info

# -------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------

def train(episodes: int = 3,
          replay_capacity: int = 20000,
          batch_size: int = 64,
          start_training_after: int = 500,
          train_every_steps: int = 8,
          epsilon_start: float = 0.9,
          epsilon_final: float = 0.05,
          epsilon_decay_steps: int = 10_000,
          weights=None, eval_mode=False):

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    if weights:
        print(f"[load] Loading weights from {weights}")
        ckpt = torch.load(weights, map_location=DEVICE)
        if isinstance(ckpt, dict) and "model" in ckpt:
            agent.q_online.load_state_dict(ckpt["model"])
        else:
            agent.q_online.load_state_dict(ckpt)
        agent.q_target.load_state_dict(agent.q_online.state_dict())
        print("[load] Model loaded successfully.")
    if eval_mode:
        agent.q_online.eval()
        epsilon_start = 0.0
        epsilon_final = 0.0
    replay_buffer = ReplayBuffer(replay_capacity)
    vision = Vision()
    env_wrapper = MergeTacticsEnv(vision=vision, tracker=BoardBenchTracker.fresh())

    global_step_counter = 0
    epsilon = epsilon_start

    for episode_index in range(episodes):
        state_vector, game_state = env_wrapper.reset()
        episode_return = 0.0
        print(f"\n--- Episode {episode_index+1}/{episodes} ---")
        save_checkpoint(agent, episode_index+1, "dqn_merge_tactics_latest.pt")
        if (episode_index + 1) % 50 == 0:
            ckpt_path = f"dqn_merge_tactics_ep{episode_index+1}.pt"
            save_checkpoint(agent, episode_index+1, ckpt_path)
        for step_index in range(10_000):
            cap = env_wrapper.compute_desired_board_for_state(game_state)
            board_count = sum(1 for s in env_wrapper.tracker.board_stars if s > 0)

            print("==============================================================================================")
            print(f"Step {step_index+1}: (cap={cap}, board_count={board_count})")

            # Epsilon linear schedule
            fraction = min(1.0, global_step_counter / max(1, epsilon_decay_steps))
            epsilon = epsilon_start + fraction * (epsilon_final - epsilon_start)

            action_mask = get_action_mask(game_state, env_wrapper.tracker, env_wrapper.last_swap_pair, cap)
            if np.count_nonzero(action_mask) == 1 and action_mask[0]:
                print("[train] Forced NoOp (only NoOp valid). Advancing passively.")
                next_state_vector, next_game_state, reward_value, done_flag, info = \
                    env_wrapper.step(0, game_state, forced_noop=True)

                state_vector, game_state = next_state_vector, next_game_state
                episode_return += reward_value
                if done_flag:
                    print(f"--- Episode {episode_index+1} Finished (passive) ---")
                    print(f"  Steps: {step_index+1}")
                    print(f"  Return: {episode_return:.2f}")
                    print(f"  Epsilon: {epsilon:.3f}")
                    print(f"  Placement: {info.get('placement')}")
                    time.sleep(2)
                    if info.get("play_again", False):
                        play_again()
                    else:
                        return_home()
                        time.sleep(1)
                        while not env.is_home_screen:
                            start_battle()
                    start_battle()
                    print("Waiting for new game to start...")
                    retry_limit = 5  # number of retry cycles (adjust as needed)
                    retry_delay = 6  # seconds between retries
                    retry_count = 0

                    while True:
                        frame = vision.capture_frame()
                        current_state = env.get_state(frame)
                        if validate_state(current_state):
                            print("New game detected. Starting next episode...")
                            break

                        time.sleep(1)
                        retry_count += 1

                        # Every 6 seconds, attempt a retry action
                        if retry_count % retry_delay == 0:
                            print(f"[wait] Still no new game after {retry_count}s. Retrying start_battle()...")
                            start_battle()

                        # Optional safety timeout
                        if retry_count >= retry_limit * retry_delay:
                            print("[wait] Timeout: forcing return_home() + start_battle()")
                            return_home()
                            time.sleep(2)
                            start_battle()
                            retry_count = 0  # reset and keep waiting
                    break  

                continue 
            action_index = agent.act(state_vector, epsilon, action_mask)

            if action_index == -1:
                print("No valid actions. Skipping step.")
                continue
            else:
                next_state_vector, next_game_state, reward_value, done_flag, info = env_wrapper.step(action_index, game_state)

            if info.get("rolled_back"):
                # Treat as if nothing happened this tick.
                state_vector, game_state = next_state_vector, next_game_state  # these equal pre-action values
                print("[train] Rolled back step; not recording transition.")
                continue

            episode_return += reward_value

            prev_state_vector = state_vector
            game_state =  next_game_state
            state_vector = next_state_vector
            if not eval_mode:
                replay_buffer.push(Transition(state_vector=prev_state_vector,
                                            action_index=action_index,
                                            reward_value=reward_value,
                                            next_state_vector=next_state_vector,
                                            done_flag=done_flag))
                

                if len(replay_buffer) >= start_training_after and global_step_counter % train_every_steps == 0:
                    print(f"  - Training step {global_step_counter}...", end="")
                    batch = replay_buffer.sample(batch_size)
                    loss_value = agent.train_step(batch)
                    print(f" Loss: {loss_value:.4f}")
            
            global_step_counter += 1
            if done_flag:
                print(f"--- Episode {episode_index+1} Finished ---")
                print(f"  Steps: {step_index+1}")
                print(f"  Return: {episode_return:.2f}")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Placement: {info.get('placement')}")
                
                # Click play again and wait for the game to reset.
                time.sleep(2)
                if info.get("play_again", False):
                    play_again()
                else:
                    return_home()
                    time.sleep(1)
                    while not env.is_home_screen:
                        start_battle()
                start_battle()
                # Poll until the game state is valid again
                print("Waiting for new game to start...")
                while True:
                    frame = vision.capture_frame()
                    current_state = env.get_state(frame)
                    if validate_state(current_state):
                        print("New game detected. Starting next episode...")
                        break
                break

    if not eval_mode:
        torch.save(agent.q_online.state_dict(), "dqn_merge_tactics.pt")
        print("[ok] Saved model to dqn_merge_tactics.pt")


# -------------------------------------------------------------------
def main():
    global DEBUG_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300, help="Number of games to play.")
    parser.add_argument("--weights", type=str, default=None, help="Path to dqn_merge_tactics.pt")
    parser.add_argument("--eval", action="store_true", help="Run in evaluation (no training) mode")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()
    DEBUG_MODE = args.debug
    print(f"Training for {args.episodes} episodes. Use --episodes to change this.")
    start_battle()
    print("Starting battle")
    train(episodes=args.episodes, weights=args.weights, eval_mode=args.eval)

if __name__ == "__main__":
    main()