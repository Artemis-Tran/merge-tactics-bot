# Merge Tactics Bot

This project is a bot for the mobile game "Merge Tactics". It uses computer vision to analyze the game state and sends touch commands via ADB to play the game automatically.

## Features

- **Game State Recognition**: The bot can read various elements from the game screen:
  - Mana
  - Player Health
  - Current Round and Phase (Deploy/Battle)
  - Cards in Hand (including cost, type, and upgradability)
  - Game Over Detection
- **Game Actions**: The bot can perform in-game actions:
  - Drag and drop units from hand/bench to the board.
  - Sell units.
  - Buy units.
  - Repeat
- **Configurable Geometry**: The screen layout (positions of cards, mana, etc.) is defined in a `geometry.json` file, making it adaptable to different screen resolutions.

## Core Components

- **`main.py`**: The main entry point for running the bot.
- **`agent.py`**: Contains the bot's decision-making logic (the "brain"). It uses the information from the `environment` to decide which actions to take via the `controller`.
- **`environment.py`**: The vision module that perceives the game world. It captures the screen, processes the image, and extracts high-level game state information like mana, health, and available cards.
- **`controller.py`**: The action module that sends commands to the game. It translates high-level actions (e.g., "drag card 0 to board position 2,3") into low-level ADB swipe commands.
- **`vision.py`**: A low-level screen capture module that interfaces with ADB to get screenshots.
- **`geometry.json`**: A configuration file that defines the regions of interest (ROIs) on the game screen for various UI elements.
- **`cards.json`**: A data file containing information about each unit in the game (cost, type, traits).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd merge-tactics-bot
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install Tesseract:**
    The project uses `pytesseract` for OCR. You need to install Tesseract on your system.
    - **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
    - **macOS (Homebrew):** `brew install tesseract`
    - **Windows:** Download from the official Tesseract repository.

4.  **Enable ADB:**
    - Enable Developer Options and USB Debugging on your Android device.
    - Install ADB on your computer and ensure your device is connected and recognized (`adb devices`).

## Usage

To run the bot:

```bash
python main.py
```

The bot will start capturing the screen, analyzing the game state, and making moves based on the logic in `agent.py`.

### Manual Control

The `controller.py` script can be used to send individual commands for testing or manual control:

```bash
# Drag a card from the first hand slot to board position (row 2, col 3)
python controller.py --from hand:0 --to board:2,3

# Sell a unit from the bench
python controller.py --from bench:1 --to hand:0
```