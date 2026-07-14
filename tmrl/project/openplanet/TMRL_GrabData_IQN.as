// TMRL IQN + lidar OpenPlanet plugin, protocol version 2.
//
// Copy this file into OpenplanetNext\Scripts.  It serves the documented
// 33-float telemetry stream on 127.0.0.1:9000 and a JSONL readiness channel on
// 127.0.0.1:9001.  Load the configured local map manually before evaluation:
// OpenPlanet exposes MapInfo.MapUid but has no documented API to safely load an
// arbitrary .Map.Gbx from a plugin.

const uint TELEMETRY_PORT = 9000;
const uint SESSION_PORT = 9001;
const string PROTOCOL_VERSION = "2";
const uint MAX_COMMAND_BYTES = 16 * 1024;

Net::Socket@ g_telemetryServer;
Net::Socket@ g_sessionServer;

void OnDestroy() {
    if (g_telemetryServer !is null) g_telemetryServer.Close();
    if (g_sessionServer !is null) g_sessionServer.Close();
}

void AppendBool(MemoryBuffer@ buf, bool value) { buf.Write(value ? 1.0f : 0.0f); }
void AppendFloat(MemoryBuffer@ buf, float value) { buf.Write(value); }
void AppendInt(MemoryBuffer@ buf, int32 value) { buf.Write(float(value)); }
void AppendVec3(MemoryBuffer@ buf, vec3 value) {
    buf.Write(value.x); buf.Write(value.y); buf.Write(value.z);
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
    if (playground is null || playground.Arena is null || playground.Arena.Players.Length == 0) return false;
    auto player = playground.Arena.Players[0];
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
        if (sock is null || !sock.IsReady()) return "";
        if (sock.Available > 0) {
            request += sock.ReadString(sock.Available);
            int newline = request.Find("\n");
            if (newline >= 0) return request.SubStr(0, newline);
            if (request.Length > MAX_COMMAND_BYTES) return "";
        }
        yield();
    }
    return "";
}

void HandleSessionClient(Net::Socket@ sock) {
    string request = ReadJsonLine(sock);
    if (request.Length == 0) { sock.Write(JsonError("invalid_or_too_large_request")); return; }
    if (request.Find("\"protocol_version\":\"" + PROTOCOL_VERSION + "\"") < 0) {
        sock.Write(JsonError("unsupported_protocol")); return;
    }
    string uid = ActiveMapUid();
    if (uid.Length == 0) { sock.Write(JsonError("no_local_map_loaded")); return; }
    if (request.Find("\"command\":\"verify_loaded_map\"") >= 0) {
        sock.Write(JsonOk(uid, false)); return;
    }
    if (request.Find("\"command\":\"confirm_ready\"") >= 0) {
        if (!IsReadyToDrive()) { sock.Write(JsonError("player_not_ready")); return; }
        sock.Write(JsonOk(uid, true)); return;
    }
    sock.Write(JsonError("unknown_command"));
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

void Main() {
    startnew(SessionServer);
    while (true) {
        @g_telemetryServer = Net::Socket();
        if (!g_telemetryServer.Listen("127.0.0.1", TELEMETRY_PORT)) { yield(120); continue; }
        Net::Socket@ sock = null;
        while (sock is null) { yield(); @sock = g_telemetryServer.Accept(); }
        MemoryBuffer@ buf = MemoryBuffer(0);
        while (sock.IsReady()) {
            CTrackMania@ app = cast<CTrackMania>(GetApp());
            CSmArenaClient@ playground = app is null ? null : cast<CSmArenaClient>(app.CurrentPlayground);
            if (playground is null || playground.Arena is null || playground.Arena.Players.Length == 0) { yield(); continue; }
            auto player = playground.Arena.Players[0];
            CSmScriptPlayer@ api = player is null ? null : cast<CSmScriptPlayer>(player.ScriptAPI);
            CSceneVehicleVisState@ vis = VehicleState::ViewingPlayerState();
            if (api is null || vis is null) { yield(); continue; }

            auto raceData = PlayerState::GetRaceData();
            bool driving = raceData.PlayerState == PlayerState::EPlayerState_Driving;
            bool finished = playground.GameTerminals.Length > 0
                && playground.GameTerminals[0].UISequence_Current == CGamePlaygroundUIConfig::EUISequence::Finish;
            buf.Seek(0, 0);
            AppendFloat(buf, driving ? float(raceData.dPlayerInfo.NumberOfCheckpointsPassed) : 0.0f);
            AppendFloat(buf, driving ? float(raceData.dPlayerInfo.CurrentLapNumber) : 0.0f);
            AppendBool(buf, finished);
            AppendFloat(buf, float(api.CurrentRaceTime));
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
            buf.Seek(0, 0);
            if (!sock.Write(buf)) break;
            yield();
        }
        sock.Close(); g_telemetryServer.Close();
    }
}
