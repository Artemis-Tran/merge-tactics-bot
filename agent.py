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
from controller import drag, Hand, Bench, Board, play_again
from vision import Vision

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
NUM_BUY_ACTIONS = 2
NUM_SELL_BOARD_ACTIONS = LOGICAL_BOARD_SLOTS
NUM_SELL_BENCH_ACTIONS = LOGICAL_BENCH_SLOTS
NUM_SWAP_ACTIONS = LOGICAL_BENCH_SLOTS * LOGICAL_BOARD_SLOTS

ACTION_SIZE = 1 + NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS + NUM_SWAP_ACTIONS

# State size: 6 ids + 6 stars + 5 ids + 5 stars + 6 ids + 6 upg + N_TRAITS + 4 scalars (mana, health, round, net_worth)
STATE_SIZE = (LOGICAL_BOARD_SLOTS * 2) + (LOGICAL_BENCH_SLOTS * 2) + (MAX_HAND_SLOTS * 2) + NUM_TRAITS + 4


# State Helper
def validate_state(state) -> bool:
    if getattr(state, "mana_conf", 0) < 50:
        return False

    # Count cards with both a label and decent confidence
    good_cards = sum(
        1 for c in getattr(state, "cards", [])[:6]
        if getattr(c, "label", None) and getattr(c, "conf", 0.0) >= 0.4
    )
    # Two solid reads is usually enough to proceed
    return good_cards >= 2

def desired_board_slots_for_round(round_num: int) -> int:
    """Per rules: desired = round + 1, capped at logical board slots."""
    return min(int(round_num or 0) + 1, LOGICAL_BOARD_SLOTS)

def logical_board_count(tracker: BoardBenchTracker) -> int:
    """How many logical board slots are currently filled (capped to 6)."""
    return min(len(tracker.list_occupied_board_indices()), LOGICAL_BOARD_SLOTS)


# -------------------------------------------------------------------
# Action Masking
# -------------------------------------------------------------------
def get_action_mask(game_state: env.GameState, tracker: BoardBenchTracker) -> np.ndarray:
    """Returns a boolean array where True indicates a valid action."""
    mask = np.zeros(ACTION_SIZE, dtype=bool)

    # Action 0: No-op is always a valid choice.
    mask[0] = True
    
    # --- Buy actions (indices 1, 2) ---
    can_buy = any(getattr(c, "label", None) for c in game_state.cards)
    if can_buy:
        if sum(1 for s in tracker.board_stars if s > 0) < LOGICAL_BOARD_SLOTS:
            mask[1] = True # Buy to board
        if sum(1 for s in tracker.bench_stars if s > 0) < LOGICAL_BENCH_SLOTS:
            mask[2] = True # Buy to bench

    # --- Sell actions ---
    num_occupied_board = len(tracker.list_occupied_board_indices())
    for i in range(LOGICAL_BOARD_SLOTS):
        if i < num_occupied_board:
            mask[1 + NUM_BUY_ACTIONS + i] = True

    for i in range(LOGICAL_BENCH_SLOTS):
        if tracker.bench_stars[i] > 0:
            mask[1 + NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + i] = True

    # --- Swap actions ---
    if game_state.phase == 'deploy':
        start_idx = 1 + NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS
        for bench_idx in range(LOGICAL_BENCH_SLOTS):
            if tracker.bench_stars[bench_idx] > 0:
                for board_idx in range(num_occupied_board):
                    action_idx = start_idx + (bench_idx * LOGICAL_BOARD_SLOTS) + board_idx
                    mask[action_idx] = True
            
    return mask


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

    @torch.no_grad()
    def act(self, state_vector: np.ndarray, epsilon: float, mask: np.ndarray) -> int:
        # If no actions are possible, do nothing (should be rare)
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

    def reset(self) -> Tuple[np.ndarray, env.GameState]:
        self.tracker = BoardBenchTracker.fresh()
        frame = self.vision.capture_frame()
        state = env.get_state(frame)
        valid_state = validate_state(state)
        while not valid_state:
            frame = self.vision.capture_frame()
            state = env.get_state(frame)
            valid_state = validate_state(state)

        # Use Roboflow to get initial state
        label, tile_index = env.get_roboflow_prediction(frame)
        if label and tile_index is not None:
            print(f"Roboflow predicted {label} at index {tile_index}")
            self.tracker.set_board_tile(tile_index, label, star_level=1)

        game_state = env.get_state(frame)
        state_vector = vectorize_state(game_state, self.tracker)
        return state_vector, game_state

    # ----------------------- helpers -----------------------
    def _enumerate_board_occupied_indices(self) -> List[int]:
        return self.tracker.list_occupied_board_indices()

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

    def _buy_from_hand(self, hand_index: int, card: env.Card, to_board: bool) -> bool:
        """Handles buying a card to board or bench, with merge logic."""
        merged = False
        if getattr(card, "upgradable", False) and card.label:
            print(f"  - Attempting merge for {card.label}")
            # This card will cause a merge. The game handles the merge itself when we buy it.
            # We just need to find which unit to upgrade in our local tracker.
            drag(Hand(hand_index), Bench(4)) # Drag to a bench slot to buy
            self.tracker.find_and_upgrade(card.label)
            merged = True
        else:
            if to_board:
                if sum(1 for s in self.tracker.board_stars if s > 0) >= LOGICAL_BOARD_SLOTS:
                    return False # logical cap reached
                board_idx = self.tracker.first_empty_board_index()
                if board_idx is not None:
                    row, col = divmod(board_idx, BOARD_COLS)
                    drag(Hand(hand_index), Board(row, col))
                    self.tracker.set_board_tile(board_idx, label=card.label, star_level=1)
            else: # to bench
                if sum(1 for s in self.tracker.bench_stars if s > 0) >= LOGICAL_BENCH_SLOTS:
                    return False
                bench_slot = self.tracker.first_empty_bench_slot()
                if bench_slot is not None:
                    drag(Hand(hand_index), Bench(bench_slot))
                    self.tracker.set_bench_slot(bench_slot, label=card.label, star_level=1)
        return merged

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


    # ----------------------- core step -----------------------
    def step(self, action_index: int, last_game_state: env.GameState) -> Tuple[np.ndarray, env.GameState, float, bool, dict]:
        """
        Executes an action and observes the next state and reward.
        """
        merged = False
        action_desc = "NoOp"

        # --- Action Decoding ---
        if action_index == 0:
            pass  # Action 0 is No-op
        else:
            action = action_index - 1 # De-offset to match original logic
            if action < NUM_BUY_ACTIONS: # Buy actions
                hand_index = self._first_nonempty_hand_index(last_game_state)
                if hand_index is not None:
                    card = last_game_state.cards[hand_index]
                    if getattr(card, "label", None) and getattr(card, "cost", 0) <= getattr(last_game_state, "mana", 0):
                        to_board = (action == 0)
                        action_desc = f"Buy {card.label} to {"Board" if to_board else "Bench"}"
                        merged = self._buy_from_hand(hand_index, card, to_board)

            elif action < NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS:
                board_slot = action - NUM_BUY_ACTIONS
                action_desc = f"Sell Board Slot {board_slot}"
                self._sell_board_slot(board_slot)

            elif action < NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS:
                bench_slot = action - (NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS)
                action_desc = f"Sell Bench Slot {bench_slot}"
                self._sell_bench_slot(bench_slot)

            else: # Swap actions
                base_idx = action - (NUM_BUY_ACTIONS + NUM_SELL_BOARD_ACTIONS + NUM_SELL_BENCH_ACTIONS)
                bench_idx = base_idx // LOGICAL_BOARD_SLOTS
                board_idx = base_idx % LOGICAL_BOARD_SLOTS
                action_desc = f"Swap Bench {bench_idx} with Board {board_idx}"
                self._swap_units(bench_idx, board_idx)

        print(f"Action: {action_desc}")

        # Observe next state
        time.sleep(0.2) # Allow UI to update
        frame = self.vision.capture_frame()
        next_game_state = env.get_state(frame)

        # --- Reward Shaping ---
        current_mana = int(getattr(next_game_state, "mana", 0.0) or 0.0)
        total_unit_value, _ = calculate_net_worth(self.tracker, 0)
        
        mana_reward = 0.01 * current_mana - 0.02
        value_reward = 0.005 * total_unit_value

        # Reward for executing a merge
        merge_reward = 0.25 if merged else 0.0
        if merged:
            print("  - MERGE DETECTED! +0.25 reward")

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # Board-fill shaping: heavily incentivize having desired units on board
        round_num = int(getattr(next_game_state, "round", 0) or 0)
        desired_slots = desired_board_slots_for_round(round_num)
        actual_slots  = logical_board_count(self.tracker)

        # Gap: how far below desired we are (never negative)
        underfill = max(0, desired_slots - actual_slots)

        # Heavier weight during battle; lighter during deploy so agent preps early
        phase = getattr(next_game_state, "phase", None)
        phase_weight = 1.75 if phase == "battle" else 0.35

        # Continuous penalty for being under desired; small nudge to not exceed desired
        board_fill_reward = -(phase_weight * underfill)
        if actual_slots > desired_slots:
            board_fill_reward -= 0.1 * (actual_slots - desired_slots)  # mild discourage

        # Extra kick exactly at deploy->battle transition:
        if getattr(last_game_state, "phase", None) == "deploy" and phase == "battle":
            # If we hit battle at/above target → big bonus; if we missed → big penalty
            board_fill_reward += (3.0 if underfill == 0 else -3.0 - 0.75 * underfill)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        step_reward = mana_reward + value_reward + merge_reward + board_fill_reward
        print(
            f"  - Reward: {step_reward:.3f} "
            f"(Mana: {mana_reward:.3f}, Value: {value_reward:.3f}, Merge: {merge_reward:.3f}, "
            f"BoardFill: {board_fill_reward:.3f} | desired={desired_slots}, actual={actual_slots}, phase={phase})"
        )

        done_flag = bool(getattr(next_game_state, "game_over", False))
        placement_value = None
        if done_flag:
            time.sleep(0.3)
            end_frame = self.vision.capture_frame()
            placement = env.get_placement(end_frame)
            placement_value = int(placement) if placement else 8
            
            terminal_reward = 0
            if placement_value == 1:
                terminal_reward = 10.0
            elif placement_value == 2:
                terminal_reward = 4.0
            elif placement_value == 3:
                terminal_reward = 2.0
            
            step_reward += terminal_reward + (0.1 * current_mana)
            print(f"  - GAME OVER! Placement: {placement_value}, Final Reward: {step_reward:.3f}")

        next_state_vector = vectorize_state(next_game_state, self.tracker)
        info = {"placement": placement_value}
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
          epsilon_decay_steps: int = 10_000):

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    replay_buffer = ReplayBuffer(replay_capacity)
    vision = Vision()
    env_wrapper = MergeTacticsEnv(vision=vision, tracker=BoardBenchTracker.fresh())

    global_step_counter = 0
    epsilon = epsilon_start

    for episode_index in range(episodes):
        state_vector, game_state = env_wrapper.reset()
        episode_return = 0.0
        print(f"\n--- Episode {episode_index+1}/{episodes} ---")

        for step_index in range(10_000):
            print(f"Step {step_index+1}:")
            # Epsilon linear schedule
            fraction = min(1.0, global_step_counter / max(1, epsilon_decay_steps))
            epsilon = epsilon_start + fraction * (epsilon_final - epsilon_start)

            action_mask = get_action_mask(game_state, env_wrapper.tracker)
            action_index = agent.act(state_vector, epsilon, action_mask)

            if action_index == -1:
                print("No valid actions. Skipping step.")
                # Get next state without taking an action
                frame = vision.capture_frame()
                next_game_state = env.get_state(frame)
                next_state_vector = vectorize_state(next_game_state, env_wrapper.tracker)
                reward_value = 0.0 # Neutral reward for waiting
                done_flag = bool(getattr(next_game_state, "game_over", False))
                info = {}
            else:
                next_state_vector, next_game_state, reward_value, done_flag, info = env_wrapper.step(action_index, game_state)

            replay_buffer.push(Transition(state_vector=state_vector,
                                          action_index=action_index,
                                          reward_value=reward_value,
                                          next_state_vector=next_state_vector,
                                          done_flag=done_flag))
            episode_return += reward_value
            state_vector, game_state = next_state_vector, next_game_state

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
                play_again()

                # Poll until the game state is valid again
                print("Waiting for new game to start...")
                while True:
                    frame = vision.capture_frame()
                    current_state = env.get_state(frame)
                    if validate_state(current_state):
                        print("New game detected. Starting next episode...")
                        break
                break

    torch.save(agent.q_online.state_dict(), "dqn_merge_tactics.pt")
    print("[ok] Saved model to dqn_merge_tactics.pt")


# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50, help="Number of games to play.")
    args = parser.parse_args()
    print(f"Training for {args.episodes} episodes. Use --episodes to change this.")
    train(episodes=args.episodes)

if __name__ == "__main__":
    main()