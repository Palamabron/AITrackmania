# Openplanet integration

For a normal installation, open Openplanet's **Plugin Manager**, install the signed
[**TrackmaniaRL Connect**](https://openplanet.dev/plugin/sac_getdata) plugin
(identifier `SAC_GetData`) and verify that it reports version **2.4.0** on
Openplanet **1.26.0 or newer**. Enable Openplanet **School
Mode**, then enter the configured local map with a visible vehicle. School Mode
deliberately prevents online play and official leaderboard submissions while the
plugin is active.

Run the complete local handshake before training:

```powershell
uv run trackmaniarl track check --config run.yaml
```

The command validates three 33-float telemetry frames, session protocol 2, the
active map UID and local-player readiness. The protocol cannot report the installed
package's signature or release number, so those two properties must be checked in
Plugin Manager. The plugin does not load maps or build geometry assets.
Its telemetry endpoint serves one client at a time, so let `track check` finish
before starting smoke or training.

The bundled `.as` source and reference manifest are a developer-reference snapshot
of the 2.4.0 source. Do not install the loose script alongside the managed plugin. The
snapshot waits until both player and vehicle-visual state exist; it never fabricates
missing orientation, wheel-slip or contact-material values with zeroes.

Official references:

- [TrackmaniaRL Connect](https://openplanet.dev/plugin/sac_getdata)
- <https://openplanet.dev/docs/school-mode>
- <https://openplanet.dev/docs/tutorials/installing-plugins>
