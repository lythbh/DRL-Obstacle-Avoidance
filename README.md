# Supervisor
Only job is to restart the simulation, needed to set ALTINO in position and run the simulation when training is complete, to make it autonomous.

---

# PPO ALTINO
Controller for ALTINO (the robot). Uses PPO (through PyTorch).

- Controller is selected in Webots.

---

# Arena setup
1. Press `+`.
2. Select `rectangle-arena`.
3. Go down to **floorSize** and change to `5 5`.
4. Mark it and press `+`.

---

# ALTINO setup
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
   - **Base nodes** → add `GPS`.
3. Expand `GPS`:
   - `translation` → set `x` to `0.03` (position on the car).
4. Add camera:
   - **Base nodes** → add `Camera`.
   - Expand `Camera`.
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

