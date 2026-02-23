# Same PC setup: WSL (server + trainer) + Windows (TrackMania + worker)

When TrackMania runs on **Windows** and you develop in **WSL**, the worker must run on **Windows** (it needs to capture the game window and send gamepad input). Run the server and trainer in WSL.

---

## Alternative: Run server, trainer, and worker all on Windows

If the worker never shows up in the server log (`Connected.` / `New client with groups ['workers']`) when server and trainer run in WSL, **run all three on Windows** to avoid WSL↔Windows networking.

**Config (single `config.json` in `C:\Users\<You>\TmrlData\config\config.json`):**

- `"LOCALHOST_WORKER": true`
- `"LOCALHOST_TRAINER": true`
- `"PUBLIC_IP_SERVER"` can stay as is; with both localhost flags true, worker and trainer will use `127.0.0.1`.
- **`"TLS": false`** for same-PC setup. If `"TLS": true`, the relay’s server subprocess must find `key.pem` and `certificate.pem` (e.g. in the default tlspyo credentials folder or in `TLS_CREDENTIALS_DIRECTORY`). If those files are missing, the subprocess crashes and **nothing listens on port 55555** (the main process still prints “listening”). With `"TLS": false`, the relay uses plain TCP and the port is opened correctly. For localhost-only use, TLS is not required.

**Steps:**

1. **Terminal 1 (PowerShell, project folder):** start the server and leave it running.
   ```powershell
   cd H:\Studia\inzynierskie\inzynierkav2\AITrackmania
   python -m tmrl --server
   ```
   You should see: `TMRL server listening on port 55555 (trainers + workers). Leave this process running.`

2. **Terminal 2 (PowerShell, same folder):** start the trainer.
   ```powershell
   python -m tmrl --trainer
   ```
   It will connect to `127.0.0.1`. You should see `server IP: 127.0.0.1` and then training logs.

3. **TrackMania:** windowed, in a map (car on track), TQC_GrabData plugin loaded.

4. **Terminal 3 (PowerShell, same folder):** start the worker.
   ```powershell
   python -m tmrl --worker
   ```
   You should see `server IP: 127.0.0.1` and, in **Terminal 1 (server)**, `Connected.` and `New client with groups ['workers'].` The trainer should then stop waiting and start training.

Use the same Python environment and project (and thus the same `config.json`) for all three. Config is loaded from **`C:\Users\<You>\TmrlData\config\config.json`** (your user profile). Ensure every process runs as the **same user** so they all see the same config.

**Important: do not run PowerShell "Run as administrator".** Run server, trainer, and worker as your normal user. If you run as Administrator, a different user profile can be used and the three processes may read different configs (or fail to find TmrlData). Running as normal user also avoids Windows Firewall treating elevated and non-elevated processes differently, which can block localhost connections.

### Port 55555 already in use (WinError 10048)

If you see **"Tylko jedno użycie każdego adresu gniazda"** or **"Only one usage of each socket address is normally permitted"** when starting the server, port 55555 is already bound by another process.

**Option A – Free the port**

1. Find what is using it (run in PowerShell; stop the tmrl server first if it crashed):
   ```powershell
   netstat -ano | findstr 55555
   ```
   The last column is the PID.

2. End that process (replace `<PID>` with the number from step 1):
   ```powershell
   taskkill /PID <PID> /F
   ```
   If it was a stuck `python -m tmrl --server`, you can start the server again.

**Option B – Use another port**

1. Open **`C:\Users\<You>\TmrlData\config\config.json`**.
2. Set `"PORT": 55559` (or any free port, e.g. 55556–55560).
3. Save the file. Server, trainer, and worker all read this config, so they will use the new port.
4. Start the server again: `python -m tmrl --server`.

### If the worker never shows up (no "New client with groups ['workers']" in server log)

1. **Same config for all:** When you start the server, it now prints the config path (e.g. `C:\Users\Kuba\TmrlData\config\config.json`). Ensure server, trainer, and worker are started from the same user so they all use that path. Run all three **without** "Run as administrator".
2. **Port and password:** In `TmrlData\config\config.json`, note `PORT` (default 55555) and `PASSWORD`. All three must use the same values. If the password does not match, tlspyo will reject the worker/trainer and you will see "The client is not connected to the Internet server" on the client side.
3. **Check port is open:** Run the server in **Terminal 1** and leave it running. On **Windows with TLS disabled**, the relay server runs in a thread (so bind errors are visible). You should see either *"Port 55555 is open and accepting connections."* (good) or a traceback if the thread failed to bind. In **Terminal 2** (with the server still running in Terminal 1) run:
   ```powershell
   Test-NetConnection -ComputerName 127.0.0.1 -Port 55555
   ```
   `TcpTestSucceeded` should be `True`. If it is `False` and the server already printed the WARNING, the relay subprocess failed to bind—often because **port 55555 is already in use**. Check with `netstat -ano | findstr 55555` (stop the server first to see if another process uses 55555), and close that process or change `PORT` in config to another port (e.g. 55559). If `"TLS": true` and TLS keys are missing, the subprocess also fails; use `"TLS": false` for same-PC.
4. **Start order:** Start **server** first, wait until you see "TMRL server listening on port ...", then start **trainer**, then **worker**.

5. **Windows and port not open:** If the server prints a WARNING that the port is not open, and you use **TLS: false**, the relay server now runs in a **thread** (not a subprocess) on Windows so any bind error is printed in the same console. Restart the server and look for a traceback. You can also run the minimal test: `python docs/debug_listen_port.py` to see if Twisted can bind to 55555 at all.

---

## 1. In WSL – get your WSL IP

From the project directory:

```bash
uv run python -m tmrl --wsl-ip
```

Or: `hostname -I | awk '{print $1}'`. Write down this IP (e.g. `172.22.123.45`). The Windows worker will use it to connect to the server.

## 2. In WSL – start server and trainer

**Terminal 1 (WSL):**
```bash
cd /mnt/h/Studia/inzynierskie/inzynierkav2/AITrackmania
uv run python -m tmrl --server
```

**Terminal 2 (WSL):**
```bash
cd /mnt/h/Studia/inzynierskie/inzynierkav2/AITrackmania
uv run python -m tmrl --trainer
```

Leave both running.

## 3. On Windows – TmrlData and config

- Install tmrl on Windows (e.g. from the same repo or `pip install tmrl`), then run once:
  ```cmd
  python -m tmrl --install
  ```
  This creates `C:\Users\<You>\TmrlData`.

- Open `C:\Users\<You>\TmrlData\config\config.json` and set:
  - **`"LOCALHOST_WORKER": false`**
  - **`"PUBLIC_IP_SERVER": "<WSL_IP>"`**  
    Replace `<WSL_IP>` with the IP from step 1 (e.g. `"172.22.123.45"`).

- Ensure the rest of the config matches your run (e.g. same `RUN_NAME`, `ENV`, `MODEL`, `ALG` as in WSL if you use a shared project). You can copy `config.json` from WSL `~/TmrlData/config/` and then only change `PUBLIC_IP_SERVER` and `LOCALHOST_WORKER`.

## 4. On Windows – start the worker

- **TrackMania:** Windowed mode, **map loaded** (you must be **in the map with the car on the track**, not in the main menu or loading screen). Load **TQC_GrabData** plugin (F3 → Developer → (Re)load plugin). You should see e.g. "Waiting for incoming connection…" in the OpenPlanet log.
- **Start order:** Either start the worker first and then in TrackMania load the map and plugin, or have the map + plugin ready and then run the worker.
- In a **Windows** terminal (PowerShell or cmd) in your project folder:
  ```cmd
  python -m tmrl --worker
  ```
- You should see either **"Connected to OpenPlanet plugin at 127.0.0.1:9000"** (then wait for "game data" while in the map) or **"Cannot connect"** (then check game + plugin + that you are in a map).
- **Click into the TrackMania window** so it has focus for the virtual gamepad.

### Car not moving? / Virtual gamepad not working

1. **ViGEmBus (Windows):** The virtual gamepad needs the ViGEmBus driver. If you see a gamepad-related error at startup, or the car never moves, install it: [ViGEmBus releases](https://github.com/ViGEm/ViGEmBus/releases) — download the installer, run it, then restart and run the worker again.
2. **Focus:** The TrackMania window must be the active (focused) window. Click inside the game after starting the worker.
3. **Controller in game:** In TrackMania go to **Settings → Input** and ensure a **controller/gamepad** is enabled for driving. If another controller is plugged in, unplug it or select the virtual one.
4. **Early training:** At the very start the policy often outputs near-zero gas; the car may only move after a few episodes.
5. **Logs:** When the worker starts you should see `Virtual gamepad (Xbox 360) initialized for control.` and after the first step `First send_control: gas=... brake=... steer=...`. If you don’t see these, the gamepad path isn’t active or reset is timing out before any step runs.

## 5. Recommended config for 1-actor TQCGRAB

For best results with a **single worker** and TQC_GrabData, use the recommended config that tunes UTD, TQC quantiles, and deeper API MLP. The training pipeline (`TorchTrainingOffline`) does not require code changes for 1 actor: it uses `max_training_steps_per_env_step` (UTD) to decide when to wait for more samples, so a higher UTD (e.g. 25) in config is enough to get more gradient steps per env step. Copy the config into place and set your WANDB key and server IP:

- **WSL:** `cp docs/config_tqcgrab_1actor_recommended.json ~/TmrlData/config/config.json`
- Then edit `~/TmrlData/config/config.json`: set `WANDB_KEY` (or use an env var) and `PUBLIC_IP_SERVER` to your WSL IP. Do not commit real API keys.

## 6. Using TQC_GrabData plugin (custom 20-float API)

If you use the **TQC_GrabData** OpenPlanet plugin (port 9000, 20 floats per frame), set in **both** WSL and Windows `TmrlData/config/config.json`:

- **`"RTGYM_INTERFACE": "TQCGRAB"`** (inside the `"ENV"` section)
- **`"USE_IMAGES": false`** (inside `"ENV"`)

The pipeline will then use `TM2020InterfaceTQC`, which expects 20 floats (e.g. checkpoint, lap, speed, position, steer, gas, brake, finished, accel, jerk, aim, steer angles, slip, isCrashed, gear) and uses the Sophy-style API-only model.

## 7. Firewall

If the Windows worker cannot connect to the server in WSL, allow inbound TCP on ports **55555**, **55556**, **55557**, **55558** for WSL (or temporarily disable the firewall for testing).

## 8. Troubleshooting: "Waiting for new samples" / trainer never gets data

If the **trainer** (WSL) stays on "Waiting for new samples" while the **worker** (Windows) logs "copying buffer for sending", the trainer’s replay buffer is empty: worker and trainer are not using the same server, or the worker’s data is not reaching it.

**"The client is not connected to the Internet server, storing message"** (from tlspyo) means the trainer or worker did not connect to the TMRL server (Relay). Typical causes: server was not running when the client started; or wrong IP (e.g. worker uses WSL IP `172.23.170.174` while the server runs on Windows — then nothing is listening on that IP). **If server, trainer, and worker all run on the same Windows PC:** in `TmrlData\config\config.json` set `"LOCALHOST_WORKER": true` and `"LOCALHOST_TRAINER": true` so both use `127.0.0.1`. Start server first, then trainer, then worker. In the server terminal you should see `Connected.` and `New client with groups ['workers'].` when the worker starts.

Do this:

1. **Same server for all three**  
   Start **server** first in WSL (`uv run python -m tmrl --server`), then trainer, then worker. Server must stay running in WSL.

2. **WSL IP and Windows config**  
   In WSL run:
   ```bash
   uv run python -m tmrl --wsl-ip
   ```
   Use the printed IP in **Windows** `TmrlData\config\config.json`:
   - `"LOCALHOST_WORKER": false`
   - `"PUBLIC_IP_SERVER": "<that WSL IP>"`  
   WSL IP can change after reboot; re-run `--wsl-ip` and update the Windows config if needed.

3. **PORT and PASSWORD must match**  
   In **both** WSL `~/TmrlData/config/config.json` and Windows `TmrlData\config\config.json` use the same:
   - `"PORT": 55555` (and same `LOCAL_PORT_*` if you changed them)
   - `"PASSWORD": "..."`  
   If they differ, the worker will not be talking to the same Relay as the trainer.

4. **Firewall**  
   Allow inbound TCP on **55555**, **55556**, **55557**, **55558** for WSL (or disable firewall briefly to test).

5. **Quick connectivity check**  
   **Start the server first** in WSL (`uv run python -m tmrl --server`), then in another WSL terminal run `ss -tlnp | grep 55555`. You should see the process listening (e.g. on `*:55555` or `0.0.0.0:55555`). If you run `ss` without the server running, you will see nothing and Windows cannot connect. From Windows, run `Test-NetConnection -ComputerName <WSL_IP> -Port 55555` in PowerShell; `TcpTestSucceeded : True` means the port is reachable.

## 9. Disabling Windows Firewall (for testing only)

If `Test-NetConnection` shows `TcpTestSucceeded : False` but `PingSucceeded : True`, the WSL host is reachable but the port is blocked. To **temporarily** allow traffic for testing (re-enable when done):

**Option A – Disable firewall for the WSL profile (recommended for testing):**

1. Open **Windows Defender Firewall** (search “Firewall” in Start).
2. Click **“Turn Windows Defender Firewall on or off”** (left side or “Allow an app through firewall”).
3. Under “Private network settings”, select **“Turn off Windows Defender Firewall”** (only for testing).
4. Click OK. Re-enable it after you finish testing.

**Option B – Allow inbound rule for WSL (PowerShell as Administrator):**

```powershell
New-NetFirewallRule -DisplayName "TMRL WSL 55555-55558" -Direction Inbound -LocalPort 55555,55556,55557,55558 -Protocol TCP -Action Allow -RemoteAddress 172.23.0.0/16
```

Adjust `172.23.0.0/16` if your WSL subnet differs (match the first two octets of your WSL IP, e.g. `172.23.170.174` → `172.23.0.0/16`).

**Option C – Disable firewall entirely (not recommended):**

```powershell
# Run PowerShell as Administrator
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
```

To re-enable: `Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True`.

## 10. WSL mirrored networking and PUBLIC_IP_SERVER

If you use **mirrored networking** in `.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
localhostForwarding=true
```

then WSL shares the **Windows host’s IP** on your LAN (e.g. **192.168.0.153**). There is no separate WSL virtual IP.

- **Important:** `PUBLIC_IP_SERVER` is the address the **worker** uses to **connect to** the server. It must be a **reachable IP/hostname**, not a bind address.
- **Do not set** `PUBLIC_IP_SERVER` to `"0.0.0.0"` in the **Windows** config. `0.0.0.0` means “listen on all interfaces” on the server; it is **not** a valid target for the worker to connect to.
- With mirrored networking, the server (running in WSL) is reached at the **same IP as your Windows machine** (e.g. **192.168.0.153**). So in **Windows** `TmrlData\config\config.json` set:
  - `"LOCALHOST_WORKER": false`
  - `"PUBLIC_IP_SERVER": "192.168.0.153"`  
  (or whatever IP your PC has on the LAN – use `ipconfig` in PowerShell and take the IPv4 address of the adapter you use, e.g. Ethernet or Wi-Fi.)
- On **WSL**, the trainer can keep `"LOCALHOST_TRAINER": true` and ignore `PUBLIC_IP_SERVER` (trainer connects to `127.0.0.1`).

## 11. Debugging connectivity step-by-step

Use this checklist when the worker cannot reach the server or the trainer never gets samples.

1. **Start the server in WSL**  
   ```bash
   uv run python -m tmrl --server
   ```  
   Leave it running. You should see no error; the process just sits there.

2. **Confirm the server is listening (WSL)**  
   In the same WSL terminal where you started the server you should see:  
   `TMRL server listening on port 55555 (trainers + workers). Leave this process running.`  
   In another WSL terminal run:
   ```bash
   ss -tlnp | grep 55555
   ```  
   You should see a line with `*:55555` or `0.0.0.0:55555`. With **mirrored** WSL networking the port may not appear in WSL’s `ss`; if the server process is running and printed the message above, and `Test-NetConnection` from Windows succeeds, the server is up.

3. **Pick the right server IP**
   - **Without** mirrored networking: in WSL run `uv run python -m tmrl --wsl-ip` and use that IP (e.g. `172.x.x.x`).
   - **With** mirrored networking: use your Windows host IP (e.g. `192.168.0.153`). In PowerShell: `ipconfig` and use the IPv4 address of the active adapter (Ethernet or Wi-Fi).

4. **Test TCP from Windows**  
   In PowerShell (replace with your server IP):
   ```powershell
   Test-NetConnection -ComputerName 192.168.0.153 -Port 55555
   ```  
   - `TcpTestSucceeded : True` → port is reachable; go to step 5.  
   - `TcpTestSucceeded : False` → firewall or wrong IP: re-check step 3, then allow ports 55555–55558 (see §9) or temporarily disable firewall.

5. **Set the worker config on Windows**  
   Edit **Windows** `C:\Users\<You>\TmrlData\config\config.json`:
   - `"LOCALHOST_WORKER": false`
   - `"PUBLIC_IP_SERVER": "<IP from step 3>"`  
   **Never** use `"0.0.0.0"` here – that is only for binding on the server side.

6. **Same PORT and PASSWORD everywhere**  
   In both WSL and Windows configs, `"PORT": 55555` and `"PASSWORD": "..."` must be identical. Otherwise the worker and trainer will not talk to the same Relay.

6b. **Confirm the worker is actually connected**  
   In the **WSL terminal where the server is running**, when the worker connects you should see:
   - `Connected.`
   - `New client with groups ['workers'].`  
   If you **never** see these after starting the worker, the worker is not reaching the server (wrong IP, firewall, or server not running). If you see `Connected.` but then `Invalid password` or the connection drops, the **PASSWORD** in Windows `TmrlData\config\config.json` does not match the server (WSL)—they must be identical.

6c. **Try 127.0.0.1 with mirrored WSL**  
   With `networkingMode=mirrored` and `localhostForwarding=true` in `.wslconfig`, from Windows, **127.0.0.1** may reach the server in WSL. In **Windows** config try `"PUBLIC_IP_SERVER": "127.0.0.1"` (keep `"LOCALHOST_WORKER": false`), restart the worker, and check the server terminal for `Connected.` and `New client with groups ['workers'].`

7. **Start order**  
   Start **server** (WSL) → **trainer** (WSL) → **worker** (Windows). Then run the worker and check that the trainer stops showing “Waiting for new samples” and that the worker logs “copying buffer for sending” and “checking for new weights” without connection errors.
