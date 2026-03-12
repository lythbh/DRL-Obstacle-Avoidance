# Environment setup
1. Create an environment with Python `3.10.19`. `python -m venv .venv`
2. Activate the environment `. .venv/bin/activate`
3. Run:
   ```bash
   pip install -r requirements.txt

---
   
# Webots setup
In Webots, click on the **Webots logo** → **Preferences** (shortcut: `Command + ,` on macOS) → **Python command** → point it to the Python executable in your environment (e.g. your virtualenv).

---

# Supervisor
Only job is to restart the simulation, needed to set ALTINO in position and run the simulation when training is complete, to make it autonomous.

---

# PPO ALTINO
Controller for ALTINO (the robot). Uses PPO (through PyTorch).

- Controller is selected in Webots.

---

# Arena setup
1. Select `rectangle-arena`.
2. Go down to **floorSize** and change to `5 5`.

---

# ALTINO setup
0. Mark  `rectangle-arena` and press `+`.
1. Go to **PROTO nodes**:  
   `webots projects -> robots -> saeon -> altino`
2. Click **Add**.
3. Expand `altino`.
4. Set a recognizable **name**.
5. Click on `altino` → set **DEF** to `ALTINO`.
6. Go down to **controller**.
7. Select the controller file (downloaded DRL controller, etc.).

---

# ALTINO sensor setup
1. Expand **children** of `ALTINO`.
2. Click `+`:
   - **Base nodes** → add `Lidar`.
3. Expand `Lidar`:
   - `translation` → set `x` to `0.03` (position on the car).
4. Click `+`:
   - **Base nodes** → add `GPS`.
5. Expand `GPS`:
   - `translation` → set `x` to `0.03` (position on the car).
6. Click `+`:
   - **Base nodes** → add `Camera`.
7. Expand `Camera`.
   - `translation` → set `x` to `0.012` and `z` to `0.06`.

---

# Supervisor setup
1. Select `rectangle-arena`.
2. Click `+` → **PROTO nodes Webots** → pick a robot from `webots projects`.
3. Expand the robot:
   - `translation` → set `x` and `y` to `2.5`.
4. Set `supervisor` to `TRUE`.
5. Controller → select `supervisor-controller`.

---

# Run
- Click **Run** to start the simulation.

