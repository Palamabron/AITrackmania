// TMRL IQN + lidar OpenPlanet plugin, protocol version 2.
//
// Copy this file into OpenplanetNext\Scripts.  It serves the documented
// 33-float telemetry stream on 127.0.0.1:9000, a JSONL readiness channel on
// 127.0.0.1:9001, and Ghost Replay Mode datagrams on 127.0.0.1:9002.
// OpenPlanet has no UDP socket API, so 9002 sends fixed-size gameTime-aligned
// datagrams over TCP (the same 144-byte layout Python tests over UDP).
// Load the configured local map manually before evaluation: OpenPlanet exposes
// MapInfo.MapUid but has no documented API to safely load an arbitrary .Map.Gbx.

const uint TELEMETRY_PORT = 9000;
const uint SESSION_PORT = 9001;
const uint GHOST_PORT = 9002;
const string PROTOCOL_VERSION = "2";
const uint MAX_COMMAND_BYTES = 16 * 1024;
const float GHOST_MAGIC = 9002.0f;
const float GHOST_PROTOCOL = 3.0f;
const float GHOST_DT_MS = 50.0f;

Net::Socket@ g_telemetryServer;
Net::Socket@ g_sessionServer;
Net::Socket@ g_ghostServer;

void OnDestroy() {
    if (g_telemetryServer !is null) g_telemetryServer.Close();
    if (g_sessionServer !is null) g_sessionServer.Close();
    if (g_ghostServer !is null) g_ghostServer.Close();
}

void AppendBool(MemoryBuffer@ buf, bool value) { buf.Write(value ? 1.0f : 0.0f); }
void AppendFloat(MemoryBuffer@ buf, float value) { buf.Write(value); }
void AppendInt(MemoryBuffer@ buf, int32 value) { buf.Write(float(value)); }
void AppendVec3(MemoryBuffer@ buf, vec3 value) {
    buf.Write(value.x); buf.Write(value.y); buf.Write(value.z);
}

int64 GameTimeNow() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    if (app is null) return 0;
    CSmArenaRulesMode@ rules = cast<CSmArenaRulesMode>(app.PlaygroundScript);
    if (rules !is null) return int64(rules.Now);
    if (app.Network is null || app.Network.PlaygroundClientScriptAPI is null) return 0;
    return int64(app.Network.PlaygroundClientScriptAPI.GameTime);
}

float RaceTimeMs(CSmScriptPlayer@ player) {
    if (player is null || player.StartTime <= 0) return 0.0f;
    int64 now = GameTimeNow();
    return now > player.StartTime ? float(now - player.StartTime) : 0.0f;
}

float GhostRaceTimeMs(CSceneVehicleVisState@ vis) {
    if (vis is null) return 0.0f;
    if (vis.RaceStartTime == 0 || vis.RaceStartTime == 0xFFFFFFFF) return 0.0f;
    int64 now = GameTimeNow();
    return now > int64(vis.RaceStartTime) ? float(now - vis.RaceStartTime) : 0.0f;
}

string ActiveMapUid() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    if (app is null) return "";
    CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.Map is null || playground.Map.MapInfo is null) return "";
    return playground.Map.MapInfo.MapUid;
}

bool IsReadyToDrive() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    if (app is null) return false;
    CSmArenaClient@ playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0) return false;
    auto player = cast<CSmPlayer>(playground.GameTerminals[0].GUIPlayer);
    return player !is null && cast<CSmScriptPlayer>(player.ScriptAPI) !is null
        && VehicleState::ViewingPlayerState() !is null;
}

string JsonError(const string &in code) {
    return "{\"status\":\"error\",\"protocol_version\":\"" + PROTOCOL_VERSION
        + "\",\"error\":\"" + code + "\"}\n";
}

string JsonOk(const string &in uid, bool ready) {
    return "{\"status\":\"ok\",\"protocol_version\":\"" + PROTOCOL_VERSION
        + "\",\"map_uid\":\"" + uid + "\",\"ready\":\""
        + (ready ? "true" : "false") + "\"}\n";
}

string ReadJsonLine(Net::Socket@ sock, uint timeoutMs = 2000) {
    string request = "";
    uint started = Time::Now;
    while (Time::Now - started < timeoutMs) {
        if (sock is null || sock.IsHungUp()) return "";
        if (!sock.IsReady()) { yield(); continue; }
        if (sock.Available() > int(MAX_COMMAND_BYTES)) return "";
        if (sock.ReadLine(request)) return request;
        yield();
    }
    return "";
}

void HandleSessionClient(Net::Socket@ sock) {
    string request = ReadJsonLine(sock);
    if (request.Length == 0) { sock.WriteRaw(JsonError("invalid_or_too_large_request")); return; }
    if (request.IndexOf("\"protocol_version\":\"" + PROTOCOL_VERSION + "\"") < 0) {
        sock.WriteRaw(JsonError("unsupported_protocol")); return;
    }
    string uid = ActiveMapUid();
    if (uid.Length == 0) { sock.WriteRaw(JsonError("no_local_map_loaded")); return; }
    if (request.IndexOf("\"command\":\"verify_loaded_map\"") >= 0) {
        sock.WriteRaw(JsonOk(uid, false)); return;
    }
    if (request.IndexOf("\"command\":\"confirm_ready\"") >= 0) {
        if (!IsReadyToDrive()) { sock.WriteRaw(JsonError("player_not_ready")); return; }
        sock.WriteRaw(JsonOk(uid, true)); return;
    }
    sock.WriteRaw(JsonError("unknown_command"));
}

void SessionServer() {
    while (true) {
        @g_sessionServer = Net::Socket();
        if (!g_sessionServer.Listen("127.0.0.1", SESSION_PORT)) { yield(120); continue; }
        while (true) {
            yield();
            Net::Socket@ sock = g_sessionServer.Accept();
            if (sock is null) continue;
            HandleSessionClient(sock);
            sock.Close();
        }
    }
}

CGameTerminal@ ActiveTerminal() {
    CTrackMania@ app = cast<CTrackMania>(GetApp());
    CSmArenaClient@ playground = app is null ? null : cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null || playground.GameTerminals.Length == 0) return null;
    return playground.GameTerminals[0];
}

bool TerminalFinished(CGameTerminal@ terminal) {
    return terminal !is null
        && terminal.UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish;
}

void AppendLiveTelemetry(MemoryBuffer@ buf, CSmScriptPlayer@ api, CSceneVehicleVisState@ vis, bool finished) {
    float raceTime = RaceTimeMs(api);
    buf.Seek(0, 0);
    AppendFloat(buf, 0.0f);
    AppendFloat(buf, 0.0f);
    AppendBool(buf, finished);
    AppendFloat(buf, raceTime);
    AppendVec3(buf, api.Position); AppendVec3(buf, api.Velocity);
    AppendVec3(buf, vis.Dir); AppendVec3(buf, vis.Up);
    AppendFloat(buf, api.Speed); AppendFloat(buf, api.EngineRpm); AppendInt(buf, api.EngineCurGear);
    AppendFloat(buf, vis.FLSlipCoef); AppendFloat(buf, vis.FRSlipCoef);
    AppendFloat(buf, vis.RLSlipCoef); AppendFloat(buf, vis.RRSlipCoef);
    AppendInt(buf, uint(vis.FLGroundContactMaterial)); AppendInt(buf, uint(vis.FRGroundContactMaterial));
    AppendInt(buf, uint(vis.RLGroundContactMaterial)); AppendInt(buf, uint(vis.RRGroundContactMaterial));
    AppendInt(buf, api.WheelsSkiddingCount); AppendInt(buf, api.FlyingDuration);
    AppendFloat(buf, api.AdherenceCoef);
    AppendFloat(buf, api.InputSteer); AppendFloat(buf, api.InputGasPedal); AppendBool(buf, api.InputIsBraking);
}

int GhostSkiddingCount(CSceneVehicleVisState@ vis) {
    int count = 0;
    if (vis.FLSlipCoef > 0.5f) count++;
    if (vis.FRSlipCoef > 0.5f) count++;
    if (vis.RLSlipCoef > 0.5f) count++;
    if (vis.RRSlipCoef > 0.5f) count++;
    return count;
}

void AppendGhostTelemetry(MemoryBuffer@ buf, CSceneVehicleVisState@ vis, bool finished, float gameTime, float raceTime) {
    buf.Seek(0, 0);
    AppendFloat(buf, GHOST_MAGIC);
    AppendFloat(buf, GHOST_PROTOCOL);
    AppendFloat(buf, gameTime);
    AppendFloat(buf, 0.0f);
    AppendFloat(buf, 0.0f);
    AppendBool(buf, finished);
    AppendFloat(buf, raceTime);
    AppendVec3(buf, vis.Position);
    AppendVec3(buf, vis.WorldVel);
    AppendVec3(buf, vis.Dir);
    AppendVec3(buf, vis.Up);
    AppendFloat(buf, vis.FrontSpeed);
    AppendFloat(buf, VehicleState::GetRPM(vis));
    AppendInt(buf, vis.CurGear);
    AppendFloat(buf, vis.FLSlipCoef); AppendFloat(buf, vis.FRSlipCoef);
    AppendFloat(buf, vis.RLSlipCoef); AppendFloat(buf, vis.RRSlipCoef);
    AppendInt(buf, uint(vis.FLGroundContactMaterial)); AppendInt(buf, uint(vis.FRGroundContactMaterial));
    AppendInt(buf, uint(vis.RLGroundContactMaterial)); AppendInt(buf, uint(vis.RRGroundContactMaterial));
    AppendInt(buf, GhostSkiddingCount(vis));
    AppendInt(buf, vis.IsGroundContact ? 0 : 1);
    AppendFloat(buf, vis.IsGroundContact ? 1.0f : 0.0f);
    AppendFloat(buf, vis.InputSteer);
    AppendFloat(buf, vis.InputGasPedal);
    AppendBool(buf, vis.InputIsBraking);
}

void TelemetryServer() {
    while (true) {
        @g_telemetryServer = Net::Socket();
        if (!g_telemetryServer.Listen("127.0.0.1", TELEMETRY_PORT)) { yield(120); continue; }
        Net::Socket@ sock = null;
        while (sock is null) { yield(); @sock = g_telemetryServer.Accept(); }
        MemoryBuffer@ buf = MemoryBuffer(0);
        while (!sock.IsHungUp()) {
            if (!sock.IsReady()) { yield(); continue; }
            CGameTerminal@ terminal = ActiveTerminal();
            if (terminal is null) { yield(); continue; }
            auto player = cast<CSmPlayer>(terminal.GUIPlayer);
            CSmScriptPlayer@ api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
            CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
            if (api is null || vis is null) { yield(); continue; }
            AppendLiveTelemetry(buf, api, vis, TerminalFinished(terminal));
            buf.Seek(0, 0);
            if (!sock.Write(buf)) break;
            yield();
        }
        sock.Close(); g_telemetryServer.Close();
    }
}

void GhostServer() {
    while (true) {
        @g_ghostServer = Net::Socket();
        if (!g_ghostServer.Listen("127.0.0.1", GHOST_PORT)) { yield(120); continue; }
        Net::Socket@ sock = null;
        while (sock is null) { yield(); @sock = g_ghostServer.Accept(); }
        MemoryBuffer@ buf = MemoryBuffer(0);
        float lastSlot = -1.0f;
        while (!sock.IsHungUp()) {
            if (!sock.IsReady()) { yield(); continue; }
            CGameTerminal@ terminal = ActiveTerminal();
            CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
            if (terminal is null || vis is null) { yield(); continue; }
            float slot = Math::Floor(float(GameTimeNow()) / GHOST_DT_MS) * GHOST_DT_MS;
            if (slot == lastSlot) { yield(); continue; }
            lastSlot = slot;
            AppendGhostTelemetry(buf, vis, TerminalFinished(terminal), slot, GhostRaceTimeMs(vis));
            buf.Seek(0, 0);
            if (!sock.Write(buf)) break;
            yield();
        }
        sock.Close(); g_ghostServer.Close();
    }
}

void Main() {
    startnew(SessionServer);
    startnew(GhostServer);
    TelemetryServer();
}
